"""Identifiability of circuit-layer connectivity weights, node gains,
baseline inputs and time constants against the behavioural evidence (P2.3).

P2.2 fitted one circuit-layer number, `gaba_a_connection_modulation_gain`,
against [Graziani2022]'s elevated-plus-maze data, and left every other
circuit-layer quantity in `scenarios.py` illustrative. P2.3's brief is to
attempt the rest - the connectivity weights and the node time constants -
against that same evidence, and to say which of them are identifiable at
all.

This module answers that question in three layers, weakest assumption
first. The layers are deliberately ordered so that the conclusions which
depend on nothing at all come before the ones that depend on how much data
there happens to be.

1. STRUCTURAL identifiability: what the model form alone allows
--------------------------------------------------------------

The behavioural observable this codebase can compare against a published
number is a drug/vehicle ratio of a STEADY-STATE readout (see
`circuit_forward.py` on why a ratio, and why a steady state: the elevated
plus maze runs 60 min after injection while the circuit's slowest tau is
2 min, so the circuit has equilibrated by a factor of 30 before the animal
is on the maze). Two exact consequences follow. Both are proved from the
model's form and then confirmed numerically, rather than inferred from a
numerical result:

  (a) **Time constants drop out entirely.** The Wilson-Cowan fixed point
      solves `r* = sigmoid(W r* + drive)`; `tau` appears only in
      `tau_i dr_i/dt = -r_i + sigmoid(...)`, i.e. only in how fast the
      approach is, never in where it lands. So d(readout ratio)/d(tau_i) is
      identically zero for every node. Not "small": zero. No amount of
      equilibrium behavioural data of any kind can constrain a tau.

  (b) **The node gains, baseline inputs and connectivity weights collapse
      into strictly fewer combinations than there are parameters.** The
      amygdala circuit is feedforward (GABA_Interneuron -| BLA -> CeA), so
      its fixed point can be written down in closed form, and when it is,
      its eight non-tau parameters enter only through four products:

          A = gain_BLA * W[BLA, GABA_Interneuron] * sigmoid_I
          B = gain_BLA * (baseline_BLA - threshold_BLA)
          C = gain_CeA * W[CeA, BLA]
          D = gain_CeA * (baseline_CeA - threshold_CeA)

      where `sigmoid_I = L(gain_I * (baseline_I - threshold_I))` is the
      interneuron's own drug-invariant rate. The drug enters as
      `A -> A * (1 + g * activation)`, so the whole observable is

          ratio(A, B, C, D, g*activation)
            = L(C * L(A*(1 + g*a) + B) + D) / L(C * L(A + B) + D)

      Eight parameters, four combinations: a four-dimensional exact
      degeneracy on top of the three dead tau directions.
      `equivalent_parameterisation()` constructs explicit, numerically very
      different parameter sets that give bit-comparable predictions, and
      `tests/test_identifiability.py` pins them.

2. PRACTICAL identifiability: what the data that exists allows
--------------------------------------------------------------

There is exactly ONE calibration-ready behavioural observation in this
codebase ([Graziani2022], via `behavioral_evidence.calibration_ready_
entries()`), and it is a single scalar with a single standard deviation.
For a scalar observation `y` with Gaussian noise, the Fisher information
matrix of any parameter vector theta is

    F = (1/sd^2) * s s^T,     s = d(prediction)/d(theta)

which is a rank-ONE matrix however many parameters theta holds. So at most
one direction in circuit-parameter space is identifiable from this dataset,
and `fisher_information_matrix()` / `fisher_spectrum()` report that as an
eigenvalue spectrum rather than as an assertion.

P2.2 spent that one direction on the drug-action gain. That was a modelling
choice, not a data-driven one, and it is defended here rather than assumed:
the gain is the only parameter in the set that is specifically about the
DRUG, so it is the only one for which a drug/vehicle contrast is the
matching kind of evidence. A connectivity weight or a time constant is a
property of the circuit, and the evidence that would pin it is
electrophysiological or connectomic, not behavioural.

3. PROFILE LIKELIHOOD: the same conclusion, reported the standard way
---------------------------------------------------------------------

The same conclusion is reported as profile likelihoods, in three regimes
that differ only in how many illustrative numbers are being treated as
known. The difference between them is the whole point:

  REGIME A - `profile_chi_square(..., free_drug_gain=False)`: the scanned
    parameter is the only free one; every other circuit number, the P2.2
    gain included, is held at its current value. Several parameters look
    identifiable here.
  REGIME B - `profile_chi_square(..., free_drug_gain=True)`: the scanned
    parameter and the drug gain are both free, and the gain is re-optimised
    at every grid point. Most of regime A's intervals open up.
  REGIME C - `profile_reduced()`: the scanned parameter and EVERY other
    circuit parameter free. Computable because the observable depends on
    the originals only through {A, B, C, D}, so profiling in that reduced
    space is the same thing as profiling with all eleven originals free.
    Nothing is localised.

Regime C is the verdict; A and B are shown because the gap between them is
the measurement of how much of an apparent fit is really the illustrative
numbers doing the work. `classify()` returns all three intervals per
parameter so that gap is visible rather than argued.

What this module does NOT do
----------------------------

It does not fit anything into `scenarios.py`, and it adds no entry to
`priors.PRIOR_SPECS`: nothing here is a production calibration, so nothing
here should pass through the guard
`tests/test_bayesian_calibration.py::test_only_evidence_backed_circuit_
layer_parameters_are_fit`. The joint models it builds
(`build_joint_diagnostic_model`) are diagnostics that measure a posterior
correlation, and they carry their own local prior specs for exactly that
reason.
"""

import numpy as np
from scipy.optimize import minimize_scalar

from .. import scenarios
from ..circuit import sigmoid
from . import behavioral_evidence as bev
from . import circuit_forward as cfwd
from . import evidence as ev
from . import forward as fwd

#: The connection [Graziani2022]'s diazepam acts on, i.e. the one key of
#: `scenarios.gaba_a_connection_modulation()`. Read from the scenario rather
#: than restated so the two cannot drift.
MODULATED_CONNECTION = next(iter(scenarios.gaba_a_connection_modulation()))

#: Chi-square(1 df) 95% quantile, the standard profile-likelihood threshold
#: (Raue et al. 2009, Bioinformatics 25(15):1923-9,
#: doi:10.1093/bioinformatics/btp358, which is also where the
#: structurally/practically non-identifiable vocabulary this module uses
#: comes from): a parameter is practically non-identifiable when its
#: profile stays below this threshold out to the edge of the searched range
#: in at least one direction.
PROFILE_THRESHOLD_CHI2 = 3.841458820694124

#: Multiplicative scan half-width for strictly-signed parameters (taus,
#: node gains, connectivity weights). 0.1x to 10x is this codebase's own
#: existing convention for "flat over the honestly-known range" when a
#: point estimate has no error bar (`priors.py`'s UNINFORMED tier). It is
#: reused here so the scan width is not a new invented quantity.
SCAN_DECADES = (0.1, 10.0)

#: Where a node's own logistic saturates. L(x) = 0.99 at x = 4.595, so an
#: additive shift of +/- 4.595/gain sweeps that node's output across
#: essentially its entire dynamic range. Used to set the scan window for
#: `baseline_input`, which (unlike a tau or a weight) can legitimately be
#: zero or negative and so has no meaningful multiplicative range.
_SIGMOID_SATURATION_ARGUMENT = float(np.log(0.99 / 0.01))

#: How many decades below its upper bound a grid whose lower bound is
#: exactly zero extends before it stops. Four decades puts the drug gain's
#: grid floor at 5e-4, two orders of magnitude below the P2.2 posterior
#: mean, which is far enough for the profile's shape near zero to be a
#: property of the likelihood rather than of the grid.
_ZERO_EDGE_LOG_DECADES = 4.0


class CircuitParameter(object):
    """One scalar circuit-layer parameter, with the machinery to set it on a
    freshly built network and the scan range the identifiability analysis
    sweeps it over.

    `nominal` is read off the scenario, never restated: every value in this
    module traces back to `scenarios.py`, which is where the illustrative
    labels live.
    """

    def __init__(self, name, kind, target, nominal, scan, scan_basis):
        self.name = name
        self.kind = kind  # "tau" | "node_gain" | "baseline_input" | "weight" | "drug_gain"
        self.target = target
        self.nominal = float(nominal)
        self.scan = (float(scan[0]), float(scan[1]))
        self.scan_basis = scan_basis

    def __repr__(self):
        return "CircuitParameter(%r, kind=%r, nominal=%g)" % (
            self.name, self.kind, self.nominal,
        )

    def set(self, network, value):
        """Apply `value` to an already-built `CircuitNetwork` in place."""
        if self.kind in ("tau", "node_gain", "baseline_input"):
            attribute = {"tau": "tau", "node_gain": "gain",
                         "baseline_input": "baseline_input"}[self.kind]
            setattr(network.nodes[network.index[self.target]], attribute, float(value))
        elif self.kind == "weight":
            post, pre = self.target
            network.connectivity[network.index[post], network.index[pre]] = float(value)
        else:
            raise TypeError(
                "%s is the drug-action gain, not a property of the network; "
                "pass it to predicted_anxiety_ratio(drug_gain=...) instead"
                % (self.name,)
            )

    def grid(self, points=41):
        """Scan grid, log-spaced for a multiplicative range (which keeps the
        nominal value at the centre of the plot) and linear otherwise.

        A range with one endpoint at exactly zero (the drug gain's, whose
        lower bound is the mechanistic null) gets zero plus a log-spaced
        tail. A linear grid would be useless there: the fitted gain is
        0.074 in a box that runs to 5, so 41 evenly spaced points would put
        the entire interesting region between the first two of them and the
        profile would be a grid artefact rather than a result.
        """
        lo, hi = self.scan
        if lo > 0.0 and hi > 0.0:
            return np.exp(np.linspace(np.log(lo), np.log(hi), points))
        if lo < 0.0 and hi < 0.0:
            return -np.exp(np.linspace(np.log(-hi), np.log(-lo), points))[::-1]
        if lo == 0.0 and hi > 0.0:
            tail = np.exp(np.linspace(np.log(hi) - _ZERO_EDGE_LOG_DECADES * np.log(10.0),
                                      np.log(hi), points - 1))
            return np.concatenate([[0.0], tail])
        return np.linspace(lo, hi, points)


def _multiplicative_scan(nominal):
    lo, hi = SCAN_DECADES[0] * nominal, SCAN_DECADES[1] * nominal
    return (lo, hi) if lo <= hi else (hi, lo)


def amygdala_circuit_parameters():
    """The eleven circuit-layer parameters of
    `scenarios.amygdala_anxiety_circuit_network()`, plus the P2.2 drug-action
    gain, in a fixed order.

    This is the only circuit in the codebase with a calibration-ready
    behavioural observation attached to it, which is why the analysis is run
    here and only here; `circuits_without_behavioural_evidence()` covers the
    other three, for which the question is settled before it is asked.
    """
    network = scenarios.amygdala_anxiety_circuit_network()
    params = []
    for node in network.nodes:
        params.append(CircuitParameter(
            "tau_%s" % node.name, "tau", node.name, node.tau,
            _multiplicative_scan(node.tau),
            "0.1x-10x the scenario value (priors.py's UNINFORMED convention)",
        ))
    for node in network.nodes:
        params.append(CircuitParameter(
            "gain_%s" % node.name, "node_gain", node.name, node.gain,
            _multiplicative_scan(node.gain),
            "0.1x-10x the scenario value (priors.py's UNINFORMED convention)",
        ))
    for node in network.nodes:
        half_width = _SIGMOID_SATURATION_ARGUMENT / node.gain
        params.append(CircuitParameter(
            "baseline_input_%s" % node.name, "baseline_input", node.name,
            node.baseline_input,
            (node.baseline_input - half_width, node.baseline_input + half_width),
            "+/- 4.595/gain, the additive shift over which this node's own "
            "logistic sweeps 1%-99% of its range",
        ))
    for post_i, post in enumerate(network.nodes):
        for pre_i, pre in enumerate(network.nodes):
            weight = network.connectivity[post_i, pre_i]
            if weight == 0.0:
                continue  # an absent connection is a structural choice, not a parameter
            params.append(CircuitParameter(
                "weight_%s__%s" % (post.name, pre.name), "weight",
                (post.name, pre.name), weight, _multiplicative_scan(weight),
                "0.1x-10x the scenario value, sign preserved (a sign flip "
                "would be a different circuit, not a different parameter)",
            ))
    params.append(CircuitParameter(
        "gaba_a_connection_modulation_gain", "drug_gain", MODULATED_CONNECTION,
        scenarios.GABA_A_CONNECTION_MODULATION_GAIN, (0.0, 5.0),
        "the P2.2 prior box (priors.PRIOR_SPECS), whose bounds come from the "
        "mechanism: 0 = no potentiation, 5 = far beyond any benzodiazepine",
    ))
    return params


AMYGDALA_PARAMETERS = amygdala_circuit_parameters()
AMYGDALA_PARAMETERS_BY_NAME = {p.name: p for p in AMYGDALA_PARAMETERS}
#: The parameters P2.3 was actually asked about: connectivity weights and
#: node time constants. Kept as a named list so the report cannot silently
#: stop covering one of them.
P2_3_TARGET_PARAMETER_NAMES = [
    p.name for p in AMYGDALA_PARAMETERS if p.kind in ("tau", "weight")
]


def build_amygdala_network(overrides=None):
    """A fresh `amygdala_anxiety_circuit_network()` with `{param_name: value}`
    applied. `scenarios` builds a new object per call, so nothing global is
    mutated."""
    network = scenarios.amygdala_anxiety_circuit_network()
    for name, value in (overrides or {}).items():
        AMYGDALA_PARAMETERS_BY_NAME[name].set(network, value)
    return network


# ---------------------------------------------------------------------------
# The observation, and the model prediction to compare with it.
# ---------------------------------------------------------------------------
#: [Graziani2022]'s drug/vehicle anxiety ratio and its source-propagated sd,
#: taken from `evidence.py` rather than restated. This is the ONLY
#: behavioural observation this codebase has; the rank-one result below is
#: a direct consequence of that "only".
OBSERVED_RATIO = ev.GABA_A_DIAZEPAM_ANXIOLYSIS_EVIDENCE["anxiety_ratio_target_mean"]
OBSERVED_RATIO_SD = ev.GABA_A_DIAZEPAM_ANXIOLYSIS_EVIDENCE["anxiety_ratio_target_sd"]


def diazepam_activation(Ki_nM=None, timepoint_min=None):
    """Diazepam's benzodiazepine-site activation at [Graziani2022]'s dose,
    through the same Phase 1 bridge + equilibrium-occupancy path P2.2 uses.

    Held fixed throughout the identifiability analysis: it is a
    RECEPTOR-layer quantity, and P2.3 is about the circuit layer. Varying it
    would confound "the circuit parameter is unidentifiable" with "the
    receptor parameter is unidentifiable", which is a different question
    that P2.2 already answered for this site.
    """
    if Ki_nM is None:
        # Read off the calibrated ligand (koff/kon = Kd), not restated: the
        # 3.6 nM [Mohler1977] figure lives in `scenarios.py` and this module
        # must not carry a second copy of it.
        diazepam = next(
            ligand for ligand in scenarios.gaba_a_benzodiazepine_ligands_calibrated()
            if ligand.name == "diazepam"
        )
        Ki_nM = diazepam.koff / diazepam.kon
    brain_nM = (
        ev.gaba_a_epm_unbound_brain_nM()
        if timepoint_min is None
        else ev.gaba_a_epm_unbound_brain_nM(timepoint_min)
    )
    return fwd.equilibrium_occupancy_from_kd(Ki_nM, brain_nM, efficacy=1.0)


def predicted_anxiety_ratio(overrides=None, drug_gain=None, activation=None,
                            baseline_rates=None):
    """The model's drug/vehicle anxiety-index ratio, via the same
    `circuit_forward.readout_ratio` P2.2's likelihood calls.

    Nothing is reimplemented for this analysis: if the production forward
    model changes, every number in this module changes with it.
    """
    if drug_gain is None:
        drug_gain = scenarios.GABA_A_CONNECTION_MODULATION_GAIN
    if activation is None:
        activation = diazepam_activation()
    network = build_amygdala_network(overrides)
    return cfwd.readout_ratio(
        network,
        scenarios.anxiety_index_readout(),
        connection_modulation={MODULATED_CONNECTION: float(drug_gain)},
        activation=activation,
        baseline_rates=baseline_rates,
    )


def _baseline_rates_for(overrides):
    """The drug-free steady state for one circuit-parameter setting. Depends
    on the circuit but not on the drug gain (at zero activation every
    modulation factor is exactly `1 + gain*0`), so profiling over the gain
    can reuse it - the same caching argument `readout_ratio` documents."""
    return cfwd.steady_state_rates(
        build_amygdala_network(overrides), {MODULATED_CONNECTION: 0.0}, activation=0.0,
    )


# ---------------------------------------------------------------------------
# Layer 1: structural identifiability, in closed form.
# ---------------------------------------------------------------------------
def structural_combinations(overrides=None):
    """The four combinations {A, B, C, D} the amygdala circuit's eight
    non-tau parameters collapse into (see the module docstring for the
    derivation).

    The collapse is exact and is a property of this circuit's feedforward
    structure, not an approximation: `GABA_Interneuron` receives no input,
    so its rate is a constant that only ever appears multiplied by
    `W[BLA, GABA_Interneuron]`, and each downstream node's gain multiplies
    its whole net input.
    """
    network = build_amygdala_network(overrides)
    node = {n.name: n for n in network.nodes}
    idx = network.index
    interneuron_rate = sigmoid(
        node["GABA_Interneuron"].baseline_input,
        gain=node["GABA_Interneuron"].gain,
        threshold=node["GABA_Interneuron"].threshold,
    )
    return {
        "A": node["BLA"].gain * network.connectivity[idx["BLA"], idx["GABA_Interneuron"]]
        * interneuron_rate,
        "B": node["BLA"].gain * (node["BLA"].baseline_input - node["BLA"].threshold),
        "C": node["CeA"].gain * network.connectivity[idx["CeA"], idx["BLA"]],
        "D": node["CeA"].gain * (node["CeA"].baseline_input - node["CeA"].threshold),
    }


def closed_form_ratio(A, B, C, D, drug_gain, activation):
    """`ratio(A, B, C, D, g*a)` written out, with no integrator involved.

    Valid only because this network is feedforward (its connectivity is
    strictly lower-triangular in the order GABA_Interneuron, BLA, CeA), so
    the fixed point can be evaluated node by node. It is used as an
    INDEPENDENT check on `predicted_anxiety_ratio`, and to make the
    four-parameter degeneracy provable rather than merely observed.
    """
    def L(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def cea(scale):
        return L(C * L(A * scale + B) + D)

    return cea(1.0 + drug_gain * activation) / cea(1.0)


def equivalent_parameterisation(seed_gains=(2.0, 0.6, 1.1)):
    """A numerically very different parameter set with identical {A,B,C,D},
    and therefore an identical prediction for every drug gain and every
    activation.

    Constructed rather than searched for: fix the three node gains to
    `seed_gains`, then solve for the baseline inputs and weights that
    preserve A, B, C and D. Returned as an `overrides` dict.
    `tests/test_identifiability.py` requires the prediction to be unchanged
    and requires at least one individual parameter to have moved by more
    than a factor of two, so this is a real relabelling and not a rounding
    difference.
    """
    network = scenarios.amygdala_anxiety_circuit_network()
    node = {n.name: n for n in network.nodes}
    target = structural_combinations()
    gain_I, gain_B, gain_C = (float(g) for g in seed_gains)

    # The interneuron's baseline is free: any (gain_I, baseline_I) pair is
    # absorbed by the weight that multiplies its rate. Keep the scenario's
    # baseline and let the weight take up the slack.
    baseline_I = node["GABA_Interneuron"].baseline_input
    interneuron_rate = sigmoid(baseline_I, gain=gain_I,
                               threshold=node["GABA_Interneuron"].threshold)
    return {
        "gain_GABA_Interneuron": gain_I,
        "gain_BLA": gain_B,
        "gain_CeA": gain_C,
        "weight_BLA__GABA_Interneuron": target["A"] / (gain_B * interneuron_rate),
        "baseline_input_BLA": target["B"] / gain_B + node["BLA"].threshold,
        "weight_CeA__BLA": target["C"] / gain_C,
        "baseline_input_CeA": target["D"] / gain_C + node["CeA"].threshold,
    }


# ---------------------------------------------------------------------------
# Layer 2: sensitivities, and the Fisher information they generate.
# ---------------------------------------------------------------------------
def sensitivity(param_name, overrides=None, drug_gain=None, activation=None,
                relative_step=1e-4):
    """d(predicted ratio)/d(parameter), by central difference.

    The step is relative to the parameter's own scale (absolute for a
    parameter whose nominal value is zero), because these parameters span
    several orders of magnitude and one absolute step would be either noise
    or a finite-size effect depending on which one it hit.
    """
    param = AMYGDALA_PARAMETERS_BY_NAME[param_name]
    if drug_gain is None:
        drug_gain = scenarios.GABA_A_CONNECTION_MODULATION_GAIN
    if activation is None:
        activation = diazepam_activation()
    base_overrides = dict(overrides or {})
    centre = base_overrides.get(param_name, param.nominal)
    if param.kind == "drug_gain":
        centre = drug_gain
    step = relative_step * abs(centre) if centre != 0.0 else relative_step

    def at(value):
        if param.kind == "drug_gain":
            return predicted_anxiety_ratio(base_overrides, value, activation)
        shifted = dict(base_overrides)
        shifted[param_name] = value
        return predicted_anxiety_ratio(shifted, drug_gain, activation)

    return (at(centre + step) - at(centre - step)) / (2.0 * step)


def sensitivity_vector(param_names=None, **kwargs):
    """The gradient of the single observable with respect to `param_names`,
    in the given order."""
    if param_names is None:
        param_names = [p.name for p in AMYGDALA_PARAMETERS]
    return np.array([sensitivity(name, **kwargs) for name in param_names])


def fisher_information_matrix(param_names=None, sd=None, **kwargs):
    """`F = s s^T / sd^2` for the one available observation.

    A rank-one matrix by construction, for any number of parameters. That
    is not a numerical accident to be checked past; it is the statement
    that one scalar datum carries one degree of freedom, and it is the
    central result of this analysis.
    """
    if sd is None:
        sd = OBSERVED_RATIO_SD
    s = sensitivity_vector(param_names, **kwargs)
    return np.outer(s, s) / (sd ** 2)


def fisher_spectrum(param_names=None, sd=None, **kwargs):
    """`(eigenvalues descending, eigenvectors as columns)` of the FIM."""
    F = fisher_information_matrix(param_names, sd=sd, **kwargs)
    values, vectors = np.linalg.eigh(F)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


# ---------------------------------------------------------------------------
# Layer 3: profile likelihood.
# ---------------------------------------------------------------------------
def chi_square(ratio, observed=None, sd=None):
    """-2 log L up to a constant, for the single Gaussian observation."""
    observed = OBSERVED_RATIO if observed is None else observed
    sd = OBSERVED_RATIO_SD if sd is None else sd
    return ((ratio - observed) / sd) ** 2


def _best_chi_square_over_drug_gain(overrides, activation, bounds=(0.0, 5.0)):
    """min over the drug gain of chi-square, at one circuit-parameter
    setting. The gain's bounds are the P2.2 prior box."""
    baseline = _baseline_rates_for(overrides)

    def objective(g):
        return chi_square(
            predicted_anxiety_ratio(overrides, g, activation, baseline_rates=baseline)
        )

    result = minimize_scalar(objective, bounds=bounds, method="bounded",
                             options={"xatol": 1e-6})
    # `minimize_scalar(method="bounded")` never evaluates the endpoints, and
    # for a monotone objective the optimum IS an endpoint, so check them.
    candidates = [(float(result.fun), float(result.x))]
    for edge in bounds:
        candidates.append((objective(edge), float(edge)))
    best = min(candidates)
    return best[0], best[1]


def profile_chi_square(param_name, grid=None, free_drug_gain=True, activation=None,
                       drug_gain=None, points=41):
    """Profile -2 log L over one parameter.

    `free_drug_gain=True` re-optimises `gaba_a_connection_modulation_gain`
    at every grid point (regime B, the honest one: the gain is a fitted
    parameter, not a known constant). `False` holds it at its P2.2 value
    (regime A, which shows how much of any apparent identifiability is
    really the other, illustrative, numbers doing the work).

    Returns `(grid, delta_chi_square, best_drug_gain)`, with the profile
    shifted so its minimum over the grid is zero, which is the form the
    delta-chi-square threshold applies to.
    """
    param = AMYGDALA_PARAMETERS_BY_NAME[param_name]
    if grid is None:
        grid = param.grid(points)
    if activation is None:
        activation = diazepam_activation()
    if drug_gain is None:
        drug_gain = scenarios.GABA_A_CONNECTION_MODULATION_GAIN

    values, gains = [], []
    for value in grid:
        if param.kind == "drug_gain":
            # Profiling the gain itself: there is nothing left to profile
            # over, in either regime, because it is the only parameter this
            # analysis ever treats as free.
            values.append(chi_square(
                predicted_anxiety_ratio(None, value, activation)))
            gains.append(float(value))
            continue
        overrides = {param_name: float(value)}
        if free_drug_gain:
            chi2, best_gain = _best_chi_square_over_drug_gain(overrides, activation)
        else:
            chi2 = chi_square(predicted_anxiety_ratio(overrides, drug_gain, activation))
            best_gain = drug_gain
        values.append(chi2)
        gains.append(best_gain)

    values = np.asarray(values, dtype=float)
    return np.asarray(grid, dtype=float), values - values.min(), np.asarray(gains)


def profile_interval(grid, delta_chi2, threshold=PROFILE_THRESHOLD_CHI2):
    """The confidence interval a profile implies, or `None` on a side where
    the profile never crosses the threshold inside the searched range.

    Returned as `(lower, upper)` with `None` for an open side, which is the
    literal statement "the data does not bound this parameter from below /
    above over the range we looked at".
    """
    grid = np.asarray(grid, dtype=float)
    delta_chi2 = np.asarray(delta_chi2, dtype=float)
    best = int(np.argmin(delta_chi2))

    def crossing(indices):
        previous = best
        for i in indices:
            if delta_chi2[i] >= threshold:
                # linear interpolation between the bracketing grid points
                x0, x1 = grid[previous], grid[i]
                y0, y1 = delta_chi2[previous], delta_chi2[i]
                if y1 == y0:
                    return float(x1)
                return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))
            previous = i
        return None

    lower = crossing(range(best - 1, -1, -1))
    upper = crossing(range(best + 1, len(grid)))
    return lower, upper


# ---------------------------------------------------------------------------
# Regime C: profiling in the REDUCED space, where the answer actually lives.
#
# Regimes A and B above both hold most of the circuit fixed, so any interval
# they produce is conditional on ten illustrative numbers. The unconditional
# question - "with every other circuit parameter free, what does the one
# observation bound?" - is answerable exactly, because the eight non-tau
# parameters enter only through {A, B, C, D}. Profiling in that four-plus-one
# dimensional space IS profiling with all the originals free, and it is cheap
# because `closed_form_ratio` needs no integrator.
# ---------------------------------------------------------------------------
REDUCED_PARAMETERS = ("A", "B", "C", "D", "gaba_a_connection_modulation_gain")


def _interval_product(*intervals):
    """Range of a product of independent intervals (attained at a corner)."""
    corners = [1.0]
    for lo, hi in intervals:
        corners = [c * e for c in corners for e in (lo, hi)]
    return min(corners), max(corners)


def reduced_parameter_box():
    """The range each of {A, B, C, D} takes as the eleven original
    parameters range over their own scan boxes.

    Propagated, not chosen: each combination is a product of independent
    factors, so its range is attained at a corner. The interneuron rate
    factor `L(gain_I * baseline_I)` is evaluated over the corners of its own
    two scan boxes for the same reason. The boxes come out very wide, which
    is the right direction for a non-identifiability claim: a profile that
    is flat over a wide box is a stronger statement than one that is flat
    over a narrow one.
    """
    p = AMYGDALA_PARAMETERS_BY_NAME
    gain_i = p["gain_GABA_Interneuron"].scan
    base_i = p["baseline_input_GABA_Interneuron"].scan
    rate_i_corners = [
        sigmoid(b, gain=g) for g in gain_i for b in base_i
    ]
    rate_i = (min(rate_i_corners), max(rate_i_corners))
    return {
        "A": _interval_product(p["gain_BLA"].scan,
                               p["weight_BLA__GABA_Interneuron"].scan, rate_i),
        "B": _interval_product(p["gain_BLA"].scan, p["baseline_input_BLA"].scan),
        "C": _interval_product(p["gain_CeA"].scan, p["weight_CeA__BLA"].scan),
        "D": _interval_product(p["gain_CeA"].scan, p["baseline_input_CeA"].scan),
        "gaba_a_connection_modulation_gain":
            p["gaba_a_connection_modulation_gain"].scan,
    }


def _reduced_chi_square(values, activation):
    return chi_square(closed_form_ratio(
        values["A"], values["B"], values["C"], values["D"],
        values["gaba_a_connection_modulation_gain"], activation,
    ))


#: Points per free coordinate in the reduced profile's inner grid search.
#: The inner problem is a 4-parameter minimisation of a non-convex
#: chi-square over a box spanning several orders of magnitude, and a plain
#: multistart L-BFGS-B fails on it in a way that MATTERS: it fails by
#: returning a chi-square that is too HIGH, which would be reported as
#: identifiability that is not there. So the inner search is an exhaustive
#: vectorised lattice followed by a local polish of the best points, not a
#: gradient method with a few starts. `closed_form_ratio` is pure numpy, so
#: 15^3 = 3375 lattice points per grid node costs milliseconds.
_REDUCED_LATTICE_POINTS = 15
#: How many of the lattice's best points get a local polish.
_REDUCED_POLISH_COUNT = 4


def _coordinate_grid(lo, hi, points):
    """A scan grid for one reduced coordinate, log-spaced where the
    coordinate is a product of positive factors (A and C span 3-4 orders of
    magnitude, so a linear grid would spend all its points in the saturated
    tail) and linear where it is a sigmoid offset (B and D)."""
    if lo > 0.0 and hi > 0.0:
        return np.exp(np.linspace(np.log(lo), np.log(hi), points))
    if lo < 0.0 and hi < 0.0:
        return -np.exp(np.linspace(np.log(-hi), np.log(-lo), points))[::-1]
    if lo == 0.0 and hi > 0.0:
        tail = np.exp(np.linspace(np.log(hi) - _ZERO_EDGE_LOG_DECADES * np.log(10.0),
                                  np.log(hi), points - 1))
        return np.concatenate([[0.0], tail])
    if lo < 0.0 and hi == 0.0:
        head = -np.exp(np.linspace(np.log(-lo),
                                   np.log(-lo) - _ZERO_EDGE_LOG_DECADES * np.log(10.0),
                                   points - 1))
        return np.concatenate([head, [0.0]])
    return np.linspace(lo, hi, points)


def profile_reduced(name, grid=None, points=41, activation=None,
                    lattice_points=_REDUCED_LATTICE_POINTS):
    """Profile -2 log L over one reduced coordinate with the other four
    free, i.e. with every original circuit parameter free.

    Returns `(grid, delta_chi_square)`. Because chi-square is bounded below
    by zero, any feasible point found is an upper bound on the profile, so
    a profile reported as flat is the conservative direction: the true
    profile can only be lower, i.e. the parameter can only be less
    identifiable than reported, never more.
    """
    from scipy.optimize import minimize

    if activation is None:
        activation = diazepam_activation()
    box = reduced_parameter_box()
    free = [k for k in REDUCED_PARAMETERS if k != name]
    if grid is None:
        grid = _coordinate_grid(box[name][0], box[name][1], points)

    bounds = [box[k] for k in free]
    lattice = np.meshgrid(
        *[_coordinate_grid(lo, hi, lattice_points) for lo, hi in bounds],
        indexing="ij",
    )
    lattice = [axis.ravel() for axis in lattice]

    values = []
    for point in grid:
        trial = {name: np.full(lattice[0].shape, float(point))}
        trial.update(dict(zip(free, lattice)))
        chi2 = _reduced_chi_square(trial, activation)
        order = np.argsort(chi2)[:_REDUCED_POLISH_COUNT]
        best = float(chi2[order[0]])
        for i in order:
            start = [float(axis[i]) for axis in lattice]

            def objective(x):
                candidate = {name: float(point)}
                candidate.update({k: float(v) for k, v in zip(free, x)})
                return _reduced_chi_square(candidate, activation)

            result = minimize(objective, start, method="L-BFGS-B", bounds=bounds)
            best = min(best, float(result.fun))
            if best < 1e-12:
                break
        values.append(best)

    values = np.asarray(values, dtype=float)
    return np.asarray(grid, dtype=float), values - values.min()


#: How wide a 95% profile interval has to be before "bounded" stops meaning
#: "calibrated". This is not a new invented threshold: it is the width of
#: this codebase's own UNINFORMED prior box (`priors.py`'s 0.1x-10x
#: convention), i.e. the span this project already uses to mean "we know
#: essentially nothing about this parameter". An interval WIDER than the
#: span that means ignorance has not narrowed anything.
#:
#: It is needed because the reduced-space profiles do produce finite bounds,
#: and they are misleading if reported bare: the bound on `A`, for instance,
#: is a 95% interval spanning five and a half orders of magnitude, whose two
#: ends are just the corners where the circuit saturates and no drug effect
#: of any size is possible. Excluding "the circuit is dead" is not the same
#: as measuring a parameter.
IGNORANCE_SPAN_RATIO = SCAN_DECADES[1] / SCAN_DECADES[0]


def interval_span_ratio(interval):
    """`max/min` in magnitude for a same-signed finite interval; `inf` for
    an open one, or one that contains or touches zero (a parameter whose
    interval straddles zero is not bounded away from any order of
    magnitude)."""
    lo, hi = interval
    if lo is None or hi is None:
        return float("inf")
    if lo == 0.0 or hi == 0.0 or lo * hi < 0.0:
        return float("inf")
    return max(abs(hi / lo), abs(lo / hi))


#: The verdicts this analysis can return, worst-determined first. The
#: structural/practical vocabulary is Raue et al. 2009's; the two extra
#: labels split cases that vocabulary lumps together and that matter here.
VERDICTS = (
    "structurally-unidentifiable",       # the observable does not depend on it at all
    "structurally-degenerate",           # it enters only inside a combination with others
    "practically-unidentifiable",        # depends on it, but the data bounds it on no side
    "one-sided-identifiable",            # the data bounds it on one side only
    "identifiable",                      # bounded on both sides at the 95% threshold
)

#: Which non-tau amygdala parameters take part in the {A,B,C,D} collapse.
#: All of them do: eight parameters, four combinations. Listed explicitly so
#: a future circuit change that breaks the collapse (any recurrent
#: connection would) shows up as a test failure rather than as silence.
STRUCTURALLY_DEGENERATE_PARAMETERS = tuple(
    p.name for p in AMYGDALA_PARAMETERS
    if p.kind in ("node_gain", "baseline_input", "weight")
)

#: Sensitivities below this are treated as zero. Set from the integrator,
#: not from taste: `circuit_forward` integrates at rtol 1e-8, so a central
#: difference of two steady states carries residual integrator noise rather
#: than a true derivative once the true derivative is smaller than that.
#: The measured separation is not marginal - the three tau sensitivities
#: come out at ~1e-7 against non-tau sensitivities of 4e-2 to 3e0, five
#: orders of magnitude apart - so the exact placement of this threshold
#: anywhere in that gap changes no verdict.
ZERO_SENSITIVITY_TOLERANCE = 1e-4


def bounds_at_searched_edge(grid, interval):
    """`(lower_at_edge, upper_at_edge)`: whether each finite endpoint of a
    profile interval sits in the outermost grid cell.

    Reported because it changes what a bound MEANS. A crossing in the
    interior is the data excluding a parameter value. A crossing in the
    outermost cell is the search box running out - "outside this box the
    circuit is saturated and no drug effect of any size is possible" -
    which is a statement about the box, and would move if the box moved.
    Both appear in this analysis and they should not be read the same way.
    """
    grid = np.asarray(grid, dtype=float)
    order = np.sort(grid)
    at_edge = []
    for side, value in enumerate(interval):
        if value is None:
            at_edge.append(False)
        elif side == 0:
            at_edge.append(bool(value <= order[1]))
        else:
            at_edge.append(bool(value >= order[-2]))
    return tuple(at_edge)


def _verdict_from_interval(interval):
    """Turn a 95% profile interval into one of `VERDICTS`.

    "Bounded on both sides" is necessary but not sufficient: the interval
    also has to be narrower than `IGNORANCE_SPAN_RATIO`, or the bound is
    excluding saturated corners rather than measuring anything.
    """
    bounded = [side is not None for side in interval]
    if not any(bounded):
        return "practically-unidentifiable"
    if not all(bounded):
        return "one-sided-identifiable"
    if interval_span_ratio(interval) >= IGNORANCE_SPAN_RATIO:
        return "practically-unidentifiable"
    return "identifiable"


def classify(param_name, grid_points=41, activation=None):
    """The full verdict for one parameter, as a dict.

    Every field is computed. Nothing in this function encodes an expected
    answer, so if the circuit or the evidence changes the verdicts change
    with it.
    """
    param = AMYGDALA_PARAMETERS_BY_NAME[param_name]
    if activation is None:
        activation = diazepam_activation()
    s = sensitivity(param_name, activation=activation)

    grid_a, profile_a, _ = profile_chi_square(
        param_name, free_drug_gain=False, activation=activation, points=grid_points)
    interval_a = profile_interval(grid_a, profile_a)
    if param.kind == "drug_gain":
        grid_b, profile_b, interval_b = grid_a, profile_a, interval_a
    else:
        grid_b, profile_b, _ = profile_chi_square(
            param_name, free_drug_gain=True, activation=activation, points=grid_points)
        interval_b = profile_interval(grid_b, profile_b)

    # Regime C. For the eight non-tau originals the answer is analytic and
    # needs no grid: each of them enters the observable only inside one of
    # {A, B, C, D}, so with the others free the profile is exactly flat and
    # the parameter is structurally degenerate. The three taus are worse
    # still (zero sensitivity). Only the drug gain has a regime-C profile
    # worth computing, and it is computed in the reduced space.
    if abs(s) < ZERO_SENSITIVITY_TOLERANCE:
        verdict = "structurally-unidentifiable"
        interval_c = (None, None)
    elif param_name in STRUCTURALLY_DEGENERATE_PARAMETERS:
        verdict = "structurally-degenerate"
        interval_c = (None, None)
    else:
        grid_c, profile_c = profile_reduced(
            param_name, points=grid_points, activation=activation)
        interval_c = profile_interval(grid_c, profile_c)
        verdict = _verdict_from_interval(interval_c)

    return {
        "parameter": param_name,
        "kind": param.kind,
        "nominal": param.nominal,
        "scan_low": param.scan[0],
        "scan_high": param.scan[1],
        "scan_basis": param.scan_basis,
        "sensitivity": float(s),
        "profile_interval_gain_fixed": interval_a,
        "profile_interval_gain_free": interval_b,
        "profile_interval_circuit_free": interval_c,
        "bounded_with_gain_fixed": all(side is not None for side in interval_a),
        "bounded_with_gain_free": all(side is not None for side in interval_b),
        "verdict": verdict,
    }


def identifiability_report(grid_points=41):
    """`classify()` for every amygdala circuit parameter, in a fixed order."""
    activation = diazepam_activation()
    return [classify(p.name, grid_points=grid_points, activation=activation)
            for p in AMYGDALA_PARAMETERS]


def reduced_report(grid_points=41):
    """The same profile analysis in the reduced {A, B, C, D, gain} space,
    i.e. over the only coordinates that COULD have been identified.

    Reported alongside the parameter-by-parameter table because it is the
    table that answers "what did the one observation actually constrain?".
    The answer, spelled out in `docs/architecture.md`'s P2.3 section, is:
    one thing, and only in one direction - the mechanistic null (drug gain
    exactly zero, which predicts a ratio of exactly 1 for EVERY circuit and
    is therefore rejected unconditionally) - plus the exclusion of the
    saturated corners where no drug effect of any size is possible.
    """
    activation = diazepam_activation()
    box = reduced_parameter_box()
    rows = []
    for name in REDUCED_PARAMETERS:
        grid, profile = profile_reduced(
            name, points=grid_points, activation=activation)
        interval = profile_interval(grid, profile)
        rows.append({
            "combination": name,
            "meaning": REDUCED_PARAMETER_MEANINGS[name],
            "box_low": box[name][0],
            "box_high": box[name][1],
            "profile_interval": interval,
            "interval_span_ratio": interval_span_ratio(interval),
            "bounds_at_searched_edge": bounds_at_searched_edge(grid, interval),
            "verdict": _verdict_from_interval(interval),
        })
    return rows


REDUCED_PARAMETER_MEANINGS = {
    "A": "gain_BLA * W[BLA, GABA_Interneuron] * interneuron rate - the "
         "GABAergic drive onto BLA, in BLA sigmoid units. The one the drug "
         "multiplies by (1 + gain*activation).",
    "B": "gain_BLA * baseline_input_BLA - BLA's tonic drive, same units.",
    "C": "gain_CeA * W[CeA, BLA] - how strongly CeA amplifies BLA.",
    "D": "gain_CeA * baseline_input_CeA - CeA's tonic drive.",
    "gaba_a_connection_modulation_gain":
        "the P2.2 drug-action gain; the only parameter here that is about "
        "the drug rather than about the circuit.",
}


# ---------------------------------------------------------------------------
# The other three circuits: the question is settled before it is asked.
# ---------------------------------------------------------------------------
#: Which `scenarios.py` circuit each behavioural readout belongs to, so the
#: "does this circuit have any behavioural observation at all?" question is
#: answered by looking at `behavioral_evidence.calibration_ready_entries()`
#: rather than by a hand-written flag that can rot.
CIRCUIT_BUILDERS = {
    "reward_circuit (mu-opioid)": scenarios.reward_circuit_network,
    "basal_ganglia_indirect_pathway (D2)": scenarios.basal_ganglia_indirect_pathway_network,
    "multi_region_mu_opioid": scenarios.multi_region_mu_opioid_network,
    "amygdala_anxiety_circuit (GABA-A)": scenarios.amygdala_anxiety_circuit_network,
}

#: The one calibration-ready behavioural entry, and the circuit it speaks
#: about. Derived from `priors.CIRCUIT_LAYER_PARAM_EVIDENCE_KEYS` in the
#: test rather than duplicated here.
CIRCUIT_OF_BEHAVIOURAL_EVIDENCE = {
    "diazepam/elevated-plus-maze/Graziani2022": "amygdala_anxiety_circuit (GABA-A)",
}

#: One-line description of the single observation, for the summary report.
GABA_A_OBSERVATION_LABEL = "%s (%s)" % (
    ev.GABA_A_DIAZEPAM_ANXIOLYSIS_EVIDENCE["evidence_key"],
    ev.GABA_A_DIAZEPAM_ANXIOLYSIS_EVIDENCE["citation"],
)


def behavioural_observation_counts():
    """{circuit label: number of calibration-ready behavioural observations}."""
    counts = {label: 0 for label in CIRCUIT_BUILDERS}
    for entry in bev.calibration_ready_entries():
        circuit = CIRCUIT_OF_BEHAVIOURAL_EVIDENCE.get(entry.key)
        if circuit is None:
            raise KeyError(
                "behavioural entry %r is calibration-ready but this module "
                "does not know which circuit it constrains; add it to "
                "CIRCUIT_OF_BEHAVIOURAL_EVIDENCE before running the analysis"
                % (entry.key,)
            )
        counts[circuit] += 1
    return counts


def count_circuit_parameters(network):
    """Number of free circuit-layer scalars in a network: tau, gain and
    baseline_input per node, plus one per non-zero connectivity entry."""
    weights = int(np.count_nonzero(np.asarray(network.connectivity, dtype=float)))
    return 3 * network.n + weights


def circuits_without_behavioural_evidence():
    """The circuits for which the identifiability question needs no analysis:
    zero observations gives a zero Fisher information matrix, so every
    parameter is unidentifiable a priori.

    Returned as a list of dicts so the report can print the parameter counts
    that stay illustrative, rather than saying "the rest".
    """
    counts = behavioural_observation_counts()
    out = []
    for label, builder in CIRCUIT_BUILDERS.items():
        if counts[label] > 0:
            continue
        out.append({
            "circuit": label,
            "behavioural_observations": 0,
            "circuit_parameters": count_circuit_parameters(builder()),
            "verdict": "structurally-unidentifiable",
            "reason": "no calibration-ready behavioural observation exists for "
                      "this circuit, so its Fisher information matrix is zero",
        })
    return out


# ---------------------------------------------------------------------------
# Diagnostic joint fits, for the posterior-correlation view of the same
# result. Deliberately NOT part of priors.PRIOR_SPECS: see module docstring.
# ---------------------------------------------------------------------------
#: Priors for the diagnostic joint models only. Each circuit parameter gets
#: a uniform prior over exactly the scan range `amygdala_circuit_parameters()`
#: assigns it, so the MCMC and the profile likelihood are asking the same
#: question over the same box and their answers can be compared directly.
def diagnostic_prior_bounds(param_name):
    return AMYGDALA_PARAMETERS_BY_NAME[param_name].scan


def build_joint_diagnostic_model(param_name, activation=None):
    """A two-parameter `pm.Model` over
    (`gaba_a_connection_modulation_gain`, `param_name`) against the single
    [Graziani2022] observation.

    The point of sampling this is not to obtain an estimate - the profile
    likelihood already says there is not one to obtain - but to SHOW the
    shape of the failure: a ridge rather than a peak, a strong posterior
    correlation between the two parameters, and a marginal for `param_name`
    barely narrower than its prior (for a tau, exactly its prior). A reader
    who does not accept a rank argument can look at the scatter plot
    instead. The measured values are in
    `results/identifiability/joint_posterior_diagnostics.csv`; the tests
    assert the qualitative shape, not a particular correlation.

    Imports PyMC lazily so that importing this module (which the fast test
    gate does) does not pay for it.
    """
    import pymc as pm
    import pytensor.tensor as pt
    from pytensor.compile.ops import as_op

    if activation is None:
        activation = diazepam_activation()
    lower, upper = diagnostic_prior_bounds(param_name)
    gain_lower, gain_upper = diagnostic_prior_bounds("gaba_a_connection_modulation_gain")

    @as_op(itypes=[pt.dvector], otypes=[pt.dvector])
    def simulate(vec):
        return np.array([
            predicted_anxiety_ratio({param_name: float(vec[1])}, float(vec[0]), activation)
        ])

    with pm.Model() as model:
        gain = pm.Uniform("gaba_a_connection_modulation_gain",
                          lower=gain_lower, upper=gain_upper)
        other = pm.Uniform(param_name, lower=lower, upper=upper)
        predicted = pm.Deterministic("predicted", simulate(pt.stack([gain, other])))
        pm.Normal(
            "diazepam_epm_anxiolysis_ratio_obs",
            mu=predicted[0], sigma=OBSERVED_RATIO_SD, observed=OBSERVED_RATIO,
        )
    return model


DIAGNOSTIC_SEED = 20260812


def run_joint_diagnostic(param_name, draws=1000, tune=1000, chains=2, cores=1,
                         progressbar=False, seed=DIAGNOSTIC_SEED):
    """Sample `build_joint_diagnostic_model` and return
    `(idata, summary, statistics)`, where `statistics` carries the two
    numbers the figure is about: the posterior correlation between the two
    parameters, and how much narrower the scanned parameter's marginal is
    than its (uniform) prior.
    """
    import arviz as az
    import pymc as pm

    model = build_joint_diagnostic_model(param_name)
    with model:
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores,
                          step=pm.DEMetropolisZ(), random_seed=seed,
                          progressbar=progressbar)

    names = ["gaba_a_connection_modulation_gain", param_name]
    summary = az.summary(idata, var_names=names)
    posterior = idata.posterior
    gain_draws = np.asarray(posterior[names[0]]).ravel()
    other_draws = np.asarray(posterior[names[1]]).ravel()
    lower, upper = diagnostic_prior_bounds(param_name)
    prior_sd = (upper - lower) / np.sqrt(12.0)  # uniform
    statistics = {
        "parameter": param_name,
        "posterior_correlation": float(np.corrcoef(gain_draws, other_draws)[0, 1]),
        "posterior_sd": float(other_draws.std(ddof=1)),
        "prior_sd": float(prior_sd),
        "posterior_sd_over_prior_sd": float(other_draws.std(ddof=1) / prior_sd),
        "max_r_hat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
    }
    return idata, summary, statistics
