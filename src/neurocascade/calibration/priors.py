"""Two-tier priors (informed vs. uninformed) for the eight free
receptor-binding parameters this pass fits, spanning all three
literature-calibrated receptor systems in `scenarios.py` (mu-opioid, D2,
GABA-A benzodiazepine site). Mirrors the two-tier scheme documented in
`/root/projects/cct-bayesian-validation/priors.py`:

  INFORMED: the `scenarios.py` citation this parameter's point estimate
  traces to ALSO reports a real standard deviation or error bar from the
  original assay (not one invented for this pass). Prior = TruncatedNormal
  (mean, sd) from that source, bounded to stay in the physically sensible
  region (positive rate constants; wide enough around the reported mean/sd
  that the bound itself is never active in practice).

  UNINFORMED: the citation reports only a single point estimate, with no
  error bar. Prior = Uniform over one order of magnitude centered on that
  point estimate (0.1x to 10x), the same "flat over the honestly-known
  range" logic `cct-bayesian-validation/priors.py` uses for its own
  literature-range-only parameters, adapted here because, unlike the CCT
  model's Table S1, none of these three point estimates comes with an
  author-reported range either, only a single number.

Two of the eight parameters below (Ki_fentanyl, Ki_flumazenil) are also the
parameters `bayesian_calibration.py`'s likelihood terms update (see
`evidence.py`); the other six are fit only in the sense that they get a
posterior at all (which, for a parameter with no matching likelihood term,
is identical to its prior). That is reported explicitly in this module and
in `docs/architecture.md`, not left implicit.

Units, matching `scenarios.py`'s calibrated sections throughout:
concentration/Ki/Kd in nM, kon in nM^-1 min^-1, koff in min^-1.
"""

PRIOR_SPECS = {
    # ------------------------------------------------------------------
    # Mu-opioid (Volpe et al. 2011, doi:10.1016/j.yrtph.2010.12.007)
    # ------------------------------------------------------------------
    # Fentanyl Ki = 1.346 nM, no error bar reported in the source text.
    # UNINFORMED: uniform over 0.1x-10x the point estimate. This is the
    # parameter the Moss et al. 2020 occupancy likelihood (evidence.py)
    # actually updates.
    "Ki_fentanyl_nM": ("uniform", 0.1346, 13.46),
    # Naloxone Ki = 1.518 +/- 0.065 nM, a real reported sd (same paper,
    # same assay as fentanyl above). INFORMED, but no matching likelihood
    # term exists in evidence.py (see docs/architecture.md): this
    # parameter's posterior will equal its prior in this pass, an honest
    # "informed prior, not yet independently updated by data" case, the
    # same category CCT's own priors.py documents for several of its
    # elicitation-only parameters.
    "Ki_naloxone_nM": ("truncnorm", 1.518, 0.065, 1.0, 2.5),

    # ------------------------------------------------------------------
    # D2 (Sykes et al. 2017, doi:10.1038/s41467-017-00716-z for
    # haloperidol; Agren et al. 2021, doi:10.3390/ijms22084078 for
    # dopamine)
    # ------------------------------------------------------------------
    # Haloperidol kon/koff are each independently measured with a real
    # reported sd (unlike the mu-opioid pair above, where koff was
    # measured but Ki was not paired with an sd). INFORMED for both;
    # together they set Kd = koff/kon, which the Kapur et al. 1997
    # likelihood term (evidence.py) constrains.
    "kon_haloperidol_nM_min": ("truncnorm", 2.13, 0.52, 0.3, 6.0),
    "koff_haloperidol_min": ("truncnorm", 0.65, 0.07, 0.05, 2.0),
    # Dopamine kon/koff are also each independently measured with a real
    # reported sd (Agren et al. 2021). INFORMED, but, like naloxone above,
    # no matching likelihood term exists: Kapur et al. 1997 validates
    # haloperidol occupancy only, not dopamine's. Posterior = prior here.
    "kon_dopamine_nM_min": ("truncnorm", 5.82, 0.738, 1.0, 12.0),
    "koff_dopamine_min": ("truncnorm", 11.82, 0.72, 2.0, 25.0),

    # ------------------------------------------------------------------
    # GABA-A benzodiazepine site (Mohler & Okada 1977,
    # doi:10.1016/0024-3205(77)90191-6 for diazepam; Mohler et al. 1981,
    # PMID 6268754 for flumazenil)
    # ------------------------------------------------------------------
    # Flumazenil Ki = 1.0 +/- 0.1 nM, a real reported sd. INFORMED, and
    # this is the parameter the Savic et al. 1991 occupancy likelihood
    # (evidence.py) updates.
    "Ki_flumazenil_nM": ("truncnorm", 1.0, 0.1, 0.3, 3.0),
    # Diazepam Ki = 3.6 +/- 0.1 nM, also a real reported sd. INFORMED, but
    # no matching likelihood term exists (Savic et al. 1991 validates
    # flumazenil occupancy only). Posterior = prior here, the same
    # informed-prior-only category as naloxone and dopamine above.
    "Ki_diazepam_nM": ("truncnorm", 3.6, 0.1, 1.0, 8.0),

    # ------------------------------------------------------------------
    # GABA-A, CIRCUIT LAYER (P2.2). The first non-receptor-binding
    # parameter this scaffold has ever fitted.
    # ------------------------------------------------------------------
    # The potentiation gain g in `gaba_a_connection_modulation()`: the
    # GABAergic interneuron -> BLA synapse's weight is scaled by
    # (1 + g * receptor_activation), so g = 0 is "diazepam does nothing to
    # this synapse" and larger g is stronger potentiation.
    #
    # UNINFORMED, and more uninformed than that label usually means here.
    # The other uninformed prior in this module (Ki_fentanyl_nM) is uniform
    # over 0.1x-10x a MEASURED point estimate. There is no measured point
    # estimate for a circuit-layer gain in any literature this project has
    # reached across Track E, P2.1 and P2.2; the only number to centre on
    # would be the hand-set illustrative 0.5 this parameter has carried
    # since v0, which is not evidence, and centring a prior on it would
    # smuggle a placeholder into the posterior.
    #
    # So the bounds are chosen from the MECHANISM instead, which is the one
    # thing here that is not arbitrary:
    #   lower 0.0 - the null. A gain of exactly zero is "diazepam does not
    #     potentiate this synapse at all", a physically meaningful and
    #     falsifiable value, not a numerical floor. Applying the 0.1x-10x
    #     convention instead would have put the lower bound at 0.05, and
    #     the [Graziani2022] posterior sits near 0.07, i.e. that bound
    #     would have been ACTIVE and the reported posterior would have been
    #     a truncation artefact. Checked before sampling, not after.
    #   upper 5.0 - at the near-saturating occupancy this exposure
    #     produces, a gain of 5 multiplies the GABAergic synapse's weight
    #     by about 5.5, far beyond any potentiation a benzodiazepine
    #     produces at a GABA-A synapse. It is a wide box, not a claim.
    # `tests/test_bayesian_calibration.py` asserts the posterior stays
    # clear of BOTH bounds; if a future scenario change pushed it against
    # one, that test fails rather than a truncated posterior being reported
    # as a fit.
    "gaba_a_connection_modulation_gain": ("uniform", 0.0, 5.0),
}

FREE_PARAM_NAMES = list(PRIOR_SPECS.keys())

#: Free parameters that are CIRCUIT-layer quantities rather than
#: receptor-binding constants, each mapped to the
#: `behavioral_evidence.BEHAVIORAL_EVIDENCE_BY_KEY` entry that calibrates
#: it. Guarded by
#: `tests/test_bayesian_calibration.py::test_only_evidence_backed_circuit_
#: layer_parameters_are_fit`, which replaced P2.1's blanket
#: "no circuit-layer parameter is fit" guard: a circuit-layer parameter may
#: now enter this scaffold, but only if it names a calibration-ready
#: behavioural dataset, and that dataset has to still be calibration-ready
#: when the test runs.
CIRCUIT_LAYER_PARAM_EVIDENCE_KEYS = {
    "gaba_a_connection_modulation_gain": "diazepam/elevated-plus-maze/Graziani2022",
}

# Parameters that get an actual likelihood-driven update in
# bayesian_calibration.py (i.e. NOT simply reproducing their prior in the
# posterior). See evidence.py for the three validation targets these two
# feed; kon_haloperidol/koff_haloperidol jointly determine the Kd the
# Kapur et al. 1997 target constrains, so both are included here even
# though the likelihood only pins down their ratio, not either value
# alone (see forward.py's identifiability note).
LIKELIHOOD_UPDATED_PARAM_NAMES = [
    "Ki_fentanyl_nM",
    "kon_haloperidol_nM_min",
    "koff_haloperidol_min",
    "Ki_flumazenil_nM",
    # Added in P2.2. `gaba_a_connection_modulation_gain` is updated by the
    # [Graziani2022] elevated-plus-maze term, and `Ki_diazepam_nM` moves out
    # of the prior-only list with it: that same term is the first likelihood
    # in this scaffold that depends on how much diazepam is actually bound,
    # so diazepam's affinity now has data behind it and not only a prior.
    # Its update is weak by construction and that is reported rather than
    # overstated - at this exposure the site is near saturation, so the
    # behavioural ratio constrains the gain sharply and the Ki barely.
    "gaba_a_connection_modulation_gain",
    "Ki_diazepam_nM",
]

# Parameters carried as informed priors only, with no likelihood term in
# this pass to further update them (see module docstring and
# docs/architecture.md).
PRIOR_ONLY_PARAM_NAMES = [
    name for name in FREE_PARAM_NAMES if name not in LIKELIHOOD_UPDATED_PARAM_NAMES
]


def scipy_frozen_dists():
    """Return {name: scipy.stats frozen distribution}, mirroring
    cct-bayesian-validation/priors.py's helper of the same name, for
    prior-predictive inspection outside of PyMC.
    """
    from scipy import stats

    dists = {}
    for name, spec in PRIOR_SPECS.items():
        kind = spec[0]
        if kind == "uniform":
            _, lo, hi = spec
            dists[name] = stats.uniform(loc=lo, scale=hi - lo)
        elif kind == "truncnorm":
            _, mean, sd, lo, hi = spec
            a, b = (lo - mean) / sd, (hi - mean) / sd
            dists[name] = stats.truncnorm(a, b, loc=mean, scale=sd)
        else:
            raise ValueError(kind)
    return dists
