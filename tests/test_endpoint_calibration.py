"""P7.3: the endpoint fits, their identifiability, and the numbers behind them.

Three groups, following P7.2's split:

* the DATA - every published value this task added, restated here so that a
  transcription error fails a test rather than a verdict;
* the RULE - what a target may and may not be built from, and what the
  information matrix says about what a contrast can identify;
* the FIT - that it runs, converges, and does not quietly remove a label.

The forward model is a synthetic `ResponseCurve` wherever the shape of the
answer is what matters, so these run in seconds and in a clean checkout.
The real 291-region curve is exercised by `scripts/run_endpoint_fits.py`
and by `tests/test_endpoint_forward.py`'s full-data section.
"""

import numpy as np
import pytest

from neurocascade import endpoints
from neurocascade.calibration import behavioral_evidence as bev
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
# The data
# ---------------------------------------------------------------------------


def test_graziani_open_field_numbers_are_the_published_ones():
    """[Graziani2022] Table 3, transcribed independently of the entry."""
    effect = bev.DIAZEPAM_OPEN_FIELD_GRAZIANI2022.effect
    assert effect.values["vehicle crossings"] == 31.1
    assert effect.dispersion["vehicle crossings"] == 2.6
    assert effect.group_sizes["vehicle crossings"] == 11
    assert effect.values["diazepam 2 mg/kg crossings"] == 43.0
    assert effect.dispersion["diazepam 2 mg/kg crossings"] == 6.1
    assert effect.group_sizes["diazepam 2 mg/kg crossings"] == 4
    assert effect.values["vehicle immobility_s"] == 57.4
    assert effect.values["diazepam 2 mg/kg immobility_s"] == 43.6


def test_the_graziani_open_field_table_sums_to_its_own_trial_duration():
    """The units check that lets this table be read as seconds without
    assuming it, the same one P2.2 ran on the EPM table next to it: centre
    plus periphery time is the stated 5 min trial in both rows."""
    effect = bev.DIAZEPAM_OPEN_FIELD_GRAZIANI2022.effect
    for group in ("vehicle", "diazepam 2 mg/kg"):
        total = (effect.values["%s centre_time_s" % group]
                 + effect.values["%s periphery_time_s" % group])
        assert total == pytest.approx(
            bev.GRAZIANI2022_OPEN_FIELD_DURATION_S, abs=6.0
        ), group


def test_sagara_numbers_are_the_published_ones():
    effect = bev.DIAZEPAM_LOCOMOTION_SAGARA2008.effect
    assert effect.values == {
        "vehicle": 153.6,
        "diazepam 0.2 mg/kg": 156.5,
        "diazepam 0.5 mg/kg": 158.3,
        "diazepam 1.0 mg/kg": 161.7,
    }
    assert effect.dispersion == {
        "vehicle": 16.6,
        "diazepam 0.2 mg/kg": 22.9,
        "diazepam 0.5 mg/kg": 24.6,
        "diazepam 1.0 mg/kg": 19.1,
    }
    assert set(effect.group_sizes.values()) == {10}
    assert effect.dispersion_kind == "SEM"
    assert bev.DIAZEPAM_LOCOMOTION_SAGARA2008.doses_mg_per_kg == (0.2, 0.5, 1.0)
    assert bev.DIAZEPAM_LOCOMOTION_SAGARA2008.timepoint_min == 15.0


def test_lustyk_numbers_are_the_published_ones_and_are_SDs():
    entry = bev.DIAZEPAM_HOT_PLATE_LUSTYK2023
    assert entry.effect.values == {
        "vehicle": 8.20,
        "diazepam 5 mg/kg": 11.40,
        "diazepam 10 mg/kg": 10.06,
    }
    assert entry.effect.dispersion == {
        "vehicle": 2.38,
        "diazepam 5 mg/kg": 4.76,
        "diazepam 10 mg/kg": 3.70,
    }
    # SD, not SEM. This is the only entry in the module reported that way
    # and the reason `DISPERSION_KINDS` records the kind rather than
    # normalising it; a caller that treated it as an SEM would be claiming
    # a width sqrt(6) too narrow.
    assert entry.effect.dispersion_kind == "SD"
    assert entry.effect.group_sizes is None
    assert bev.LUSTYK2023_MIN_GROUP_SIZE == 6


def test_yassen_logistic_is_the_published_one_and_reproduces_its_own_paper():
    """Two checks on one transcription.

    The first restates the published parameters. The second is the
    arithmetic that caught the source's own Discussion out: alpha and beta
    give 1.30% and 3.84% at 4 and 5 ng/mL, where that Discussion claims
    0.4% and 1.2% - which are the values at 3 and 4 ng/mL. The respiratory
    half of the same sentence reproduces exactly from its own parameters,
    which is what makes the antinociceptive half a misprint rather than a
    different convention.
    """
    assert bev.YASSEN2008_ALPHA == -8.77
    assert bev.YASSEN2008_BETA_PER_NG_PER_ML == 1.11
    p4, p5 = bev.yassen2008_antinociception_probability([4.0, 5.0])
    assert p4 == pytest.approx(0.0130, abs=5e-4)
    assert p5 == pytest.approx(0.0384, abs=5e-4)
    p3 = float(bev.yassen2008_antinociception_probability(3.0))
    assert p3 == pytest.approx(0.0043, abs=5e-4)
    # The respiratory parameters, which DO reproduce the same sentence.
    resp = 1.0 / (1.0 + np.exp(-(-5.86 + 0.930 * np.array([4.0, 5.0]))))
    assert resp[0] == pytest.approx(0.105, abs=2e-3)
    assert resp[1] == pytest.approx(0.230, abs=2e-3)


def test_the_published_analgesia_curve_is_far_steeper_than_any_occupancy_curve():
    """The measured reason the analgesia endpoint is not fitted.

    A one-site occupancy curve spans 10% to 90% over an 81-fold
    concentration range, for every drug and every Kd, because occupancy is
    C/(C+Kd). [Yassen2008]'s curve does it in 1.67-fold. No monotone
    function of fractional occupancy can be that steep.
    """
    assert bev.HILL_ONE_STEEPNESS == 81.0
    assert bev.YASSEN2008_STEEPNESS == pytest.approx(1.669, abs=1e-3)
    assert bev.YASSEN2008_STEEPNESS < bev.HILL_ONE_STEEPNESS / 40.0
    # Derived independently of the module's own helper.
    c10 = (np.log(0.1 / 0.9) + 8.77) / 1.11
    c90 = (np.log(0.9 / 0.1) + 8.77) / 1.11
    assert c90 / c10 == pytest.approx(bev.YASSEN2008_STEEPNESS, rel=1e-9)


def test_obert_and_kissin_numbers_are_the_published_ones():
    assert bev.OBERT2025_RORR_MIN_BY_DOSE_MG_PER_KG == {
        0.0125: 13.0, 0.025: 15.0, 0.050: 26.0, 0.100: 50.0,
    }
    assert bev.OBERT2025_RORR_IQR_MIN_BY_DOSE_MG_PER_KG[0.100] == (45.0, 53.0)
    assert bev.KISSIN1983_FENTANYL_RR_ED50_MG_PER_KG == 0.023
    assert bev.KISSIN1983_FENTANYL_RR_ED50_FIDUCIAL_LIMITS_MG_PER_KG == (
        0.020, 0.026
    )
    # An IQR is not a dispersion this module can express, and no conversion
    # was invented for it.
    assert bev.FENTANYL_RIGHTING_REFLEX_OBERT2025.effect.dispersion is None


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def test_every_endpoint_is_covered_exactly_once():
    coverage = bev.endpoint_coverage()
    assert set(coverage) == set(endpoints.ENDPOINTS)
    assert len(coverage) == 7


def test_every_coverage_record_says_something_specific():
    for key, record in bev.ENDPOINT_COVERAGE.items():
        assert len(record["record"]) > 400, key
        assert record["verdict"] in bev.ENDPOINT_COVERAGE_VERDICTS


def test_no_endpoint_is_recorded_as_fitted_because_none_was():
    """The label guard, in the direction that matters.

    P7.3 removed no illustrative label. If a later pass fits a gain it must
    change this test deliberately, in the same commit that removes the
    label - which is rule 3 of this codebase expressed as an assertion.
    """
    verdicts = {k: v["verdict"] for k, v in bev.ENDPOINT_COVERAGE.items()}
    assert "fitted" not in verdicts.values()


def test_the_whole_brain_gains_stay_out_of_the_calibrated_register():
    """`priors.CIRCUIT_LAYER_PARAM_EVIDENCE_KEYS` is the list of
    circuit-layer parameters this project HAS calibrated. Nothing P7.3
    sampled belongs there, because nothing P7.3 sampled came back
    identified."""
    for name in ec.WHOLE_BRAIN_PRIOR_SPECS:
        assert name not in priors.PRIOR_SPECS
        assert name not in priors.CIRCUIT_LAYER_PARAM_EVIDENCE_KEYS
        assert name not in priors.LIKELIHOOD_UPDATED_PARAM_NAMES


def test_the_p2_2_gain_is_untouched():
    """P2.2's fitted amygdala gain is a different parameter and P7.3 did
    not move it."""
    assert priors.PRIOR_SPECS["gaba_a_connection_modulation_gain"] == (
        "uniform", 0.0, 5.0
    )
    assert (priors.CIRCUIT_LAYER_PARAM_EVIDENCE_KEYS
            ["gaba_a_connection_modulation_gain"]
            == "diazepam/elevated-plus-maze/Graziani2022")


def test_a_target_must_declare_its_polarity():
    with pytest.raises(ValueError):
        ec.EndpointTarget(
            key="x", endpoint_key="anxiety", evidence_key="k", measure="m",
            gene="Gabra1", mechanism="connection_modulation",
            levels=((1.0, 10.0, 0.9, 0.1),), exposure_timepoint_min=60.0,
            Ki_nM=3.6, efficacy=1.0, measure_direction=0,
        )


def test_an_unknown_endpoint_is_refused():
    with pytest.raises(ValueError):
        ec.EndpointTarget(
            key="x", endpoint_key="telepathy", evidence_key="k", measure="m",
            gene="Gabra1", mechanism="connection_modulation",
            levels=((1.0, 10.0, 0.9, 0.1),), exposure_timepoint_min=60.0,
            Ki_nM=3.6, efficacy=1.0, measure_direction=1,
        )


def test_a_measure_that_opposes_its_endpoint_is_inverted_once():
    target = ec.EndpointTarget(
        key="x", endpoint_key="anxiety", evidence_key="k", measure="m",
        gene="Gabra1", mechanism="connection_modulation",
        levels=((1.0, 10.0, 0.5, 0.1),), exposure_timepoint_min=60.0,
        Ki_nM=3.6, efficacy=1.0, measure_direction=-1,
    )
    ratios, sds = target.endpoint_ratio()
    assert ratios[0] == pytest.approx(2.0)
    assert sds[0] == pytest.approx(0.1 / 0.25)


def test_every_target_reaches_a_receptor_driving_concentration():
    """The criterion that excludes every fentanyl entry from the fits, and
    the reason it is computed from `drug_pk` rather than written down: no
    rat brain unbound fraction for fentanyl exists, so its Kp is a
    total-brain ratio and must not drive occupancy."""
    for target in ec.TARGETS:
        assert target.entry.can_drive_receptor, target.key
    fentanyl = bev.BEHAVIORAL_EVIDENCE_BY_KEY["fentanyl/tail-flick/Yassen2008"]
    assert fentanyl.is_bridgeable
    assert not fentanyl.can_drive_receptor
    assert fentanyl.receptor_drive_status()[0] == "total-brain-only"


def test_every_target_names_a_real_entry_and_a_real_endpoint():
    for target in ec.TARGETS:
        assert target.evidence_key in bev.BEHAVIORAL_EVIDENCE_BY_KEY
        assert target.endpoint_key in endpoints.ENDPOINTS
        assert target.entry.can_drive_receptor
        assert target.n_levels >= 1


def test_the_graded_targets_have_more_than_one_level_and_the_others_do_not():
    levels = {t.key: t.n_levels for t in ec.TARGETS}
    assert levels["locomotion/diazepam/Sagara2008"] == 3
    assert levels["analgesia/diazepam/Lustyk2023"] == 2
    assert levels["anxiety/diazepam/Graziani2022"] == 1


def test_a_higher_dose_reaches_a_higher_exposure():
    """The Phase 1 bridge is applied per dose, not once - the whole point
    of a graded series - so the exposures must be ordered like the doses."""
    for key in ("locomotion/diazepam/Sagara2008",
                "analgesia/diazepam/Lustyk2023"):
        target = ec.TARGETS_BY_KEY[key]
        exposures = [level[1] for level in target.levels]
        assert exposures == sorted(exposures), key
        assert exposures[0] < exposures[-1], key


def test_one_contrast_gives_a_rank_one_information_matrix():
    """P2.3's argument, at a different scale and for a different reason.

    A single Gaussian observation has `F = s s^T / sd^2` whatever `s` is,
    so a two-point contrast cannot separate a circuit gain from an unknown
    behavioural scale. This is what forces P7.3 to fit ratios rather than
    a scale, and it is measured rather than asserted.
    """
    curve = synthetic_curve(["anxiety"])
    target = ec.TARGETS_BY_KEY["anxiety/diazepam/Graziani2022"]
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
    target = ec.TARGETS_BY_KEY["locomotion/diazepam/Sagara2008"]
    fisher = ec.information_matrix(curve, target, gain=2.5)
    eigenvalues = np.linalg.eigvalsh(fisher)
    assert abs(eigenvalues[0]) > 1e-12 * abs(eigenvalues[1])


def test_the_verdicts_are_computed_from_the_curve_not_written_down():
    """A model that moves the endpoint the wrong way must be reported as
    wrong-direction, and one that moves it the right way but not far
    enough as beyond-a-bound. Both are checked on synthetic curves whose
    direction is known by construction, so a wrong rule fails here rather
    than in a results file."""
    target = ec.TARGETS_BY_KEY["anxiety/diazepam/Graziani2022"]
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


def test_the_best_residual_is_reported_in_units_of_the_published_sd():
    target = ec.TARGETS_BY_KEY["anxiety/diazepam/Graziani2022"]
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


def test_the_fit_runs_and_converges_on_a_synthetic_curve():
    """A smoke test at a cheap sampler configuration, in the shape P2.2's
    own smoke tests take: short chains, real sampler, real diagnostics."""
    import pandas as pd

    target = ec.TARGETS_BY_KEY["locomotion/diazepam/Sagara2008"]
    curve = synthetic_curve(["locomotion"], slope=-0.05)
    idata, summary = ec.run_fit(
        target, curve, draws=200, tune=200, chains=2
    )
    assert "gaba_a_whole_brain_modulation_gain" in summary.index
    r_hat = float(pd.to_numeric(summary["r_hat"], errors="coerce").max())
    assert np.isfinite(r_hat)
    posterior = idata.posterior["gaba_a_whole_brain_modulation_gain"].values
    assert posterior.min() >= 0.0 and posterior.max() <= 5.0


def test_a_null_series_pushes_the_gain_down_rather_than_up():
    """What [Sagara2008] is fitted FOR. The model predicts a monotone
    decrease in the locomotion index; the published series shows none; so
    the posterior must sit low in its box, which is a one-sided
    identification rather than a value."""
    target = ec.TARGETS_BY_KEY["locomotion/diazepam/Sagara2008"]
    curve = synthetic_curve(["locomotion"], slope=-0.2)
    idata, summary = ec.run_fit(target, curve, draws=400, tune=400, chains=2)
    mean = float(summary["mean"].iloc[0])
    prior_mean = 2.5
    assert mean < prior_mean
