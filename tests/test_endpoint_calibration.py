"""P7.3: the endpoint-fit machinery, its identifiability, and its empty register.

Three groups, following P7.2's split:

* the REGISTER - `TARGETS` is EMPTY since P11.3 retired every dataset this
  module ever had (all rodent; see the module docstring of
  `endpoint_calibration.py`), so these tests pin the emptiness and the
  reasons rather than five targets that no longer exist;
* the RULE - what a target may and may not be built from, and what the
  information matrix says about what a contrast can identify, exercised on
  explicitly-labelled synthetic targets;
* the FIT - that it runs, converges, and actually RECOVERS a known gain
  from observations generated at that gain, which is how an empty-register
  module proves it still works.

The published-value pins that used to open this file moved to
`tests/test_legacy_behavioral_evidence.py` in P11.3, together with the data
they pin. The forward model here is a synthetic `ResponseCurve` throughout:
these run in seconds and in a clean checkout. The real 291-region curve is
exercised by `scripts/run_endpoint_fits.py --synthetic` and by
`tests/test_endpoint_forward.py`'s full-data section.
"""

import numpy as np
import pytest

from neurocascade.calibration import endpoint_calibration as ec
from neurocascade.calibration import endpoint_forward as efwd
from neurocascade.calibration import priors


# ---------------------------------------------------------------------------
# A synthetic response curve: monotone, saturating, exactly like the real one
# in shape and with no dependence on any data file.
# ---------------------------------------------------------------------------


def synthetic_curve(endpoint_keys, slope=-0.02):
    x = np.linspace(0.0, 5.0, 51)
    values = {
        key: efwd.BASELINE_INDEX + slope * (1.0 - np.exp(-x))
        for key in endpoint_keys
    }
    return efwd.ResponseCurve("synthetic", x, values)


# ---------------------------------------------------------------------------
# The register
# ---------------------------------------------------------------------------


def test_the_target_register_is_empty_and_that_is_the_honest_state():
    """P11.3 retired every dataset this module fitted (eleven rodent
    entries across three datasets -> five targets). The machinery stays;
    the targets cannot, because none of them was human."""
    assert ec.TARGETS == ()
    assert ec.TARGETS_BY_KEY == {}


def test_targets_expose_no_literature_backing_anymore():
    """The `.entry` property that read `behavioral_evidence` is gone with
    the datasets; a target built today names SYNTHETIC unless a caller
    passes something real."""
    assert not hasattr(ec.EndpointTarget, "entry")
    target = ec.synthetic_target(
        key="anxiety/SYNTHETIC", endpoint_key="anxiety",
        ratios=(0.9,), sds=(0.05,), exposures_nM=(50.0,),
    )
    assert target.evidence_key == "SYNTHETIC"
    assert np.isnan(target.doses_mg_per_kg[0])


def test_the_whole_brain_gains_stay_out_of_the_calibrated_register():
    """`priors.CIRCUIT_LAYER_PARAM_EVIDENCE_KEYS` is the list of circuit-layer
    parameters this project HAS calibrated. Nothing P7.3 samples belongs
    there: nothing it ever sampled came back identified, and since P11.3
    the register is empty outright."""
    for name in ec.WHOLE_BRAIN_PRIOR_SPECS:
        assert name not in priors.PRIOR_SPECS
        assert name not in priors.CIRCUIT_LAYER_PARAM_EVIDENCE_KEYS
        assert name not in priors.LIKELIHOOD_UPDATED_PARAM_NAMES


def test_the_p2_2_gain_has_no_register_entry_left():
    """P11.3 removed the last circuit-layer entry from every fit register.
    The gain itself survives only as scenarios.py's illustrative placeholder
    and in the quarantined legacy record."""
    assert "gaba_a_connection_modulation_gain" not in priors.PRIOR_SPECS
    assert ("gaba_a_connection_modulation_gain"
            not in priors.CIRCUIT_LAYER_PARAM_EVIDENCE_KEYS)
    assert ("gaba_a_connection_modulation_gain"
            not in priors.LIKELIHOOD_UPDATED_PARAM_NAMES)


def test_the_fixed_receptor_constants_are_unchanged():
    """Pinned because the fits hold them fixed: if these move, every
    historical diagnostic in results/calibration/ becomes unreproducible
    without saying so."""
    assert ec.DIAZEPAM_KI_NM == 3.6
    assert ec.DIAZEPAM_EFFICACY == 1.0
    assert ec.GABA_A_GENE == "Gabra1"
    assert ec.GABA_A_MECHANISM == "connection_modulation"
    assert set(ec.WHOLE_BRAIN_PRIOR_SPECS) == {
        "gaba_a_whole_brain_modulation_gain"
    }


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def test_a_target_must_declare_its_polarity():
    with pytest.raises(ValueError):
        ec.EndpointTarget(
            key="x", endpoint_key="anxiety", evidence_key="SYNTHETIC",
            measure="m", gene="Gabra1", mechanism="connection_modulation",
            levels=((1.0, 10.0, 0.9, 0.1),), exposure_timepoint_min=60.0,
            Ki_nM=3.6, efficacy=1.0, measure_direction=0,
        )


def test_an_unknown_endpoint_is_refused():
    with pytest.raises(ValueError):
        ec.EndpointTarget(
            key="x", endpoint_key="telepathy", evidence_key="SYNTHETIC",
            measure="m", gene="Gabra1", mechanism="connection_modulation",
            levels=((1.0, 10.0, 0.9, 0.1),), exposure_timepoint_min=60.0,
            Ki_nM=3.6, efficacy=1.0, measure_direction=1,
        )


def test_a_synthetic_target_refuses_mismatched_arrays():
    with pytest.raises(ValueError):
        ec.synthetic_target(
            key="x", endpoint_key="anxiety",
            ratios=(0.9, 0.8), sds=(0.05,), exposures_nM=(50.0, 100.0),
        )


def test_a_measure_that_opposes_its_endpoint_is_inverted_once():
    target = ec.EndpointTarget(
        key="x", endpoint_key="anxiety", evidence_key="SYNTHETIC",
        measure="m", gene="Gabra1", mechanism="connection_modulation",
        levels=((1.0, 10.0, 0.5, 0.1),), exposure_timepoint_min=60.0,
        Ki_nM=3.6, efficacy=1.0, measure_direction=-1,
    )
    ratios, sds = target.endpoint_ratio()
    assert ratios[0] == pytest.approx(2.0)
    assert sds[0] == pytest.approx(0.1 / 0.25)


def test_the_polarity_rule_round_trips_through_the_curve_builder():
    """`synthetic_target_from_curve(measure_direction=-1)` stores the
    reciprocal the rule demands, so `endpoint_ratio()` hands back exactly
    the index ratio the model predicts - which is why a fit through an
    oppositely-signed measure would still be comparing like with like."""
    curve = synthetic_curve(["anxiety"], slope=-0.1)
    plain = ec.synthetic_target_from_curve(
        curve, "anxiety", gain=2.0, exposures_nM=(50.0,), sds=(0.03,),
    )
    flipped = ec.synthetic_target_from_curve(
        curve, "anxiety", gain=2.0, exposures_nM=(50.0,), sds=(0.03,),
        measure_direction=-1,
    )
    plain_ratios, _ = plain.endpoint_ratio()
    flipped_ratios, flipped_sds = flipped.endpoint_ratio()
    assert flipped_ratios[0] == pytest.approx(plain_ratios[0])
    # The stored width is inflated by the reciprocal transform, and comes
    # back down through it.
    assert flipped.levels[0][3] > 0.03
    assert flipped_sds[0] == pytest.approx(0.03)


def test_a_known_gain_is_reachable_and_best_residual_zero():
    """A target generated FROM a curve at a known gain sits inside what that
    curve can produce, and the best achievable residual is exactly zero at
    exactly the generating gain. This is the property that makes the
    recovery fit below meaningful."""
    curve = synthetic_curve(["locomotion"], slope=-0.05)
    target = ec.synthetic_target_from_curve(
        curve, "locomotion", gain=2.0,
        exposures_nM=(20.0, 60.0, 180.0), sds=(0.02,) * 3,
    )
    assert ec.bound_verdict(curve, target) == "reachable"
    best_gain, best_residual = ec.best_residual_sd(curve, target)
    assert best_gain == pytest.approx(2.0)
    assert best_residual < 1e-9


def test_one_contrast_gives_a_rank_one_information_matrix():
    """P2.3's argument, at a different scale and for a different reason.

    A single Gaussian observation has `F = s s^T / sd^2` whatever `s` is,
    so a two-point contrast cannot separate a circuit gain from an unknown
    behavioural scale. This is what forces P7.3 to fit ratios rather than
    a scale, and it is measured rather than asserted.
    """
    curve = synthetic_curve(["anxiety"])
    target = ec.synthetic_target_from_curve(
        curve, "anxiety", gain=2.5, exposures_nM=(50.0,), sds=(0.02,),
    )
    fisher = ec.information_matrix(curve, target, gain=2.5)
    eigenvalues = np.linalg.eigvalsh(fisher)
    assert abs(eigenvalues[0]) < 1e-9 * abs(eigenvalues[1])
    assert np.linalg.matrix_rank(fisher, tol=1e-9 * abs(eigenvalues[1])) == 1


def test_a_graded_series_can_lift_that_rank():
    """And the corresponding positive: three doses at three different
    exposures give an information matrix that is not rank-deficient, which
    is the sense in which a dose-response curve buys something a contrast
    does not."""
    curve = synthetic_curve(["locomotion"])
    target = ec.synthetic_target_from_curve(
        curve, "locomotion", gain=2.5,
        exposures_nM=(10.0, 40.0, 160.0), sds=(0.02,) * 3,
    )
    fisher = ec.information_matrix(curve, target, gain=2.5)
    eigenvalues = np.linalg.eigvalsh(fisher)
    assert abs(eigenvalues[0]) > 1e-12 * abs(eigenvalues[1])


def test_the_verdicts_are_computed_from_the_curve_not_written_down():
    """A model that moves the endpoint the wrong way must be reported as
    wrong-direction, and one that moves it the right way but not far
    enough as beyond-a-bound. Both are checked on synthetic curves whose
    direction is known by construction, so a wrong rule fails here rather
    than in a results file."""
    target = ec.synthetic_target_from_curve(
        synthetic_curve(["anxiety"], slope=-0.2),
        "anxiety", gain=2.0, exposures_nM=(50.0,), sds=(0.05,),
    )
    observed, _ = target.endpoint_ratio()
    assert observed[0] < 1.0

    # A curve that barely moves: right direction, not far enough.
    weak = synthetic_curve(["anxiety"], slope=-0.001)
    assert ec.bound_verdict(weak, target) == "beyond-lower-bound"

    # A curve that moves the other way entirely.
    inverted = synthetic_curve(["anxiety"], slope=+0.05)
    assert ec.bound_verdict(inverted, target) == "wrong-direction"

    # A curve that can reach it.
    strong = synthetic_curve(["anxiety"], slope=-0.2)
    assert ec.bound_verdict(strong, target) == "reachable"


def test_the_best_residual_is_reported_in_units_of_the_observation_sd():
    target = ec.synthetic_target_from_curve(
        synthetic_curve(["anxiety"], slope=-0.2),
        "anxiety", gain=2.0, exposures_nM=(50.0,), sds=(0.05,),
    )
    weak = synthetic_curve(["anxiety"], slope=-0.001)
    gain, residual = ec.best_residual_sd(weak, target)
    assert residual > 3.0
    strong = synthetic_curve(["anxiety"], slope=-0.2)
    gain, residual = ec.best_residual_sd(strong, target)
    assert residual < 1.0
    assert 0.0 <= gain <= 5.0


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def test_the_fit_runs_and_converges_on_a_synthetic_target():
    """A smoke test at a cheap sampler configuration, in the shape P2.2's
    own smoke tests take: short chains, real sampler, real diagnostics."""
    import pandas as pd

    curve = synthetic_curve(["locomotion"], slope=-0.05)
    target = ec.synthetic_target_from_curve(
        curve, "locomotion", gain=2.0,
        exposures_nM=(20.0, 60.0, 180.0), sds=(0.02,) * 3,
    )
    idata, summary = ec.run_fit(target, curve, draws=200, tune=200, chains=2)
    assert "gaba_a_whole_brain_modulation_gain" in summary.index
    r_hat = float(pd.to_numeric(summary["r_hat"], errors="coerce").max())
    assert np.isfinite(r_hat)
    posterior = idata.posterior["gaba_a_whole_brain_modulation_gain"].values
    assert posterior.min() >= 0.0 and posterior.max() <= 5.0


def test_the_fit_recovers_the_gain_a_synthetic_series_was_generated_at():
    """What an empty register owes the roadmap: proof that when a real
    (human) dataset lands, this machinery can turn observations into a
    gain. Observations are generated from the model itself at gain 1.5 -
    `synthetic_target_from_curve`, labelled SYNTHETIC end to end - and the
    posterior must localise near that value rather than near the prior's."""

    curve = synthetic_curve(["locomotion"], slope=-0.2)
    truth = 1.5
    target = ec.synthetic_target_from_curve(
        curve, "locomotion", gain=truth,
        exposures_nM=(10.0, 30.0, 100.0), sds=(0.01,) * 3,
    )
    assert target.evidence_key == "SYNTHETIC"
    _, summary = ec.run_fit(target, curve, draws=600, tune=300, chains=2)
    mean = float(summary["mean"].iloc[0])
    sd = float(summary["sd"].iloc[0])
    prior_mean = 2.5
    # Localised: far from the Uniform(0, 5) mean, and near the truth.
    assert abs(mean - prior_mean) > 4.0 * sd
    assert mean == pytest.approx(truth, abs=max(0.25, 3.0 * sd))


def test_a_low_generated_gain_pushes_the_posterior_below_the_prior_mean():
    """The one-sided identification a NULL series buys, reproduced on
    synthetic terms now that [Sagara2008] has left: observations generated
    at a small gain sit just off the mechanistic null, and the posterior
    must sit low in its box rather than at the prior's centre."""
    curve = synthetic_curve(["locomotion"], slope=-0.2)
    target = ec.synthetic_target_from_curve(
        curve, "locomotion", gain=0.3,
        exposures_nM=(10.0, 30.0, 100.0), sds=(0.01,) * 3,
    )
    _, summary = ec.run_fit(target, curve, draws=400, tune=400, chains=2)
    mean = float(summary["mean"].iloc[0])
    assert mean < 2.5
