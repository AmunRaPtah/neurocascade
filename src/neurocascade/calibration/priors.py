"""Two-tier priors (informed vs. uninformed) for the seven free
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

Two of the seven parameters below (Ki_fentanyl, Ki_flumazenil) are also the
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
    # informed-prior-only category as naloxone and dopamine above. Between
    # P2.2 and P11.3 the [Graziani2022] behavioural-ratio likelihood did
    # weakly update it; that term was rodent-derived and retired with its
    # dataset, so diazepam's affinity is prior-only again.
    "Ki_diazepam_nM": ("truncnorm", 3.6, 0.1, 1.0, 8.0),

    # ------------------------------------------------------------------
    # GABA-A, CIRCUIT LAYER: deliberately ABSENT since P11.3.
    # ------------------------------------------------------------------
    # Between P2.2 and P11.3 this register carried
    # `gaba_a_connection_modulation_gain` = Uniform(0.0, 5.0), because that
    # gain was fitted against [Graziani2022]'s Wistar-rat elevated-plus-maze
    # data and so had a posterior to place a prior under. The species pivot
    # retired that fit (the observation is rat behaviour; see
    # `scenarios.GABA_A_CONNECTION_MODULATION_GAIN`, restored to its
    # illustrative placeholder) and with it the parameter's entry here: a
    # prior spec for a quantity nothing fits would advertise an intent this
    # scaffold cannot honestly carry until HUMAN behavioural data exists.
    # The machinery that would fit a circuit-layer gain again is kept warm
    # on synthetic observations - `bayesian_calibration.build_circuit_gain_
    # model` and `identifiability.py`'s synthetic scalar - so re-adding the
    # entry when such data arrives is a one-line change plus the same-commit
    # calibration docs/roadmap.md's rules require.
}

FREE_PARAM_NAMES = list(PRIOR_SPECS.keys())

#: Free parameters that are CIRCUIT-layer quantities rather than
#: receptor-binding constants, each mapped to the behavioural-evidence entry
#: that calibrates it. EMPTY since P11.3: the one entry it ever held was the
#: rodent-derived gain, retired with its dataset. This register is the
#: load-bearing side of
#: `tests/test_bayesian_calibration.py::test_no_circuit_layer_parameter_
#: claims_to_be_calibrated`, which since P11.3 enforces P2.1's original
#: blanket rule again - no circuit-layer parameter enters the fit register -
#: and will fail the moment anything is added here without naming human
#: evidence for it in the same change.
CIRCUIT_LAYER_PARAM_EVIDENCE_KEYS = {}

# Parameters that get an actual likelihood-driven update in
# bayesian_calibration.py (i.e. NOT simply reproducing their prior in the
# posterior). See evidence.py for the three validation targets these feed;
# kon_haloperidol/koff_haloperidol jointly determine the Kd the
# Kapur et al. 1997 target constrains, so both are included here even
# though the likelihood only pins down their ratio, not either value
# alone (see forward.py's identifiability note).
LIKELIHOOD_UPDATED_PARAM_NAMES = [
    "Ki_fentanyl_nM",
    "kon_haloperidol_nM_min",
    "koff_haloperidol_min",
    "Ki_flumazenil_nM",
    # Between P2.2 and P11.3 this list also carried
    # `gaba_a_connection_modulation_gain` and `Ki_diazepam_nM`: the
    # [Graziani2022] elevated-plus-maze term updated the gain directly and
    # diazepam's affinity weakly through it. Both left with the term when
    # the rodent dataset retired.
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
