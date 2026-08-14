"""P7.3: the whole-brain forward model, from a gain to an endpoint index.

Split the way P7.2's tests are: the ALGEBRA (properties that must hold for
any network, checked on the committed fifteen-region fixtures so a clean
checkout runs them), the NUMERICS (the interpolation shortcut, checked
against a re-integration rather than assumed), and the FULL-DATA checks
that need the gitignored 291-region cache.
"""

import numpy as np
import pytest

from neurocascade import connectome, endpoints, stability
from neurocascade.calibration import endpoint_forward as efwd
from neurocascade.connectome import fetch, sources


@pytest.fixture(scope="module")
def fixture_model():
    ontology = connectome.load_fixture_ontology()
    strength, voxels = connectome.load_fixture_build_inputs(ontology=ontology)
    model, provenance = efwd.build_whole_brain_model(
        "Gabra1", "connection_modulation", name="test/fixture",
        inputs=(ontology, strength, voxels),
    )
    return model, provenance


@pytest.fixture(scope="module")
def drive_model():
    ontology = connectome.load_fixture_ontology()
    strength, voxels = connectome.load_fixture_build_inputs(ontology=ontology)
    model, _ = efwd.build_whole_brain_model(
        "Oprm1", "drive_coupling", name="test/fixture-drive",
        inputs=(ontology, strength, voxels),
    )
    return model


# ---------------------------------------------------------------------------
# The algebra
# ---------------------------------------------------------------------------


def test_only_the_product_of_gain_and_activation_matters(drive_model):
    """The reduction the whole MCMC shortcut rests on.

    A gain of -3 at 40% activation and a gain of -1.2 at full activation
    are the same drug as far as the circuit is concerned, because both
    enter only as `gain * activation`. If this ever stopped being true the
    `ResponseCurve` table would be indexed by the wrong thing, so it is
    checked rather than asserted in a docstring.
    """
    a = drive_model.rates(-3.0 * 0.4)
    b = drive_model.rates(-1.2 * 1.0)
    c = drive_model.rates(-6.0 * 0.2)
    assert np.allclose(a, b, atol=1e-9)
    assert np.allclose(a, c, atol=1e-9)


def test_zero_gain_returns_the_drug_free_state(fixture_model, drive_model):
    model, _ = fixture_model
    assert np.allclose(model.rates(0.0), model.baseline_rates, atol=1e-9)
    assert np.allclose(drive_model.rates(0.0), drive_model.baseline_rates,
                       atol=1e-9)


def test_the_drug_free_index_is_exactly_one_half_for_every_endpoint(
        fixture_model):
    """The property that keeps every drug/vehicle ratio off a near-zero
    denominator, and the reason `directed_activity_index` exists at all.

    It is exact, not approximate: the resolved weights satisfy
    `sum |w| == 1` and P3.4 balances every node to r = 0.5.
    """
    model, _ = fixture_model
    for key in endpoints.ENDPOINTS:
        index = model.index(key, model.baseline_rates)
        assert index == pytest.approx(efwd.BASELINE_INDEX, abs=1e-6), key


def test_the_index_is_the_score_plus_the_negative_mass(fixture_model):
    """Two independent readings of the same number: the definition in
    `directed_activity_index` and the affine identity it implies."""
    model, _ = fixture_model
    rates = model.rates(1.5)
    names = [node.name for node in model.network.nodes]
    by_node = dict(zip(names, rates))
    for key in endpoints.ENDPOINTS:
        resolved = model.resolved_endpoint(key)
        weights = resolved.readout.weights
        if not weights:
            continue
        negative_mass = sum(-w for w in weights.values() if w < 0)
        score = sum(w * by_node[n] for n, w in weights.items())
        assert model.index(key, rates) == pytest.approx(
            score + negative_mass, abs=1e-9
        ), key


def test_the_index_stays_inside_the_unit_interval(fixture_model):
    model, _ = fixture_model
    for x in (-8.0, -1.0, 0.0, 1.0, 5.0):
        rates = model.rates(x)
        for key in endpoints.ENDPOINTS:
            index = model.index(key, rates)
            assert -1e-9 <= index <= 1.0 + 1e-9, (key, x, index)


def test_a_region_with_no_density_has_to_be_named(fixture_model):
    """`density.as_coupling` raises on a NaN rather than emit a zero,
    because "not measured" is not "no target here". This class takes the
    same position one layer up: a node that has neither a density nor a
    place on the unmeasured list is a construction error."""
    model, _ = fixture_model
    densities = dict(model.densities)
    dropped = sorted(densities)[0]
    del densities[dropped]
    with pytest.raises(ValueError) as excinfo:
        efwd.WholeBrainDrugModel(
            network=model.network, ontology=model.ontology,
            densities=densities, mechanism=model.mechanism, name="bad",
        )
    assert dropped in str(excinfo.value)


def test_an_unknown_mechanism_is_refused(fixture_model):
    model, _ = fixture_model
    with pytest.raises(ValueError):
        efwd.WholeBrainDrugModel(
            network=model.network, ontology=model.ontology,
            densities=model.densities, mechanism="hand-waving", name="bad",
        )


def test_the_two_mechanisms_move_the_brain_in_opposite_directions(
        fixture_model, drive_model):
    """A sanity check on the sign convention that a whole task's verdicts
    depend on: at a POSITIVE x, potentiating inhibition lowers the mean
    rate, while an inhibitory drive coupling needs a NEGATIVE x to do the
    same thing. Getting this backwards would invert every
    `wrong-direction` verdict in P7.3."""
    model, _ = fixture_model
    assert model.rates(2.0).mean() < model.baseline_rates.mean()
    assert drive_model.rates(-2.0).mean() < drive_model.baseline_rates.mean()
    assert drive_model.rates(2.0).mean() > drive_model.baseline_rates.mean()


# ---------------------------------------------------------------------------
# The numerics
# ---------------------------------------------------------------------------


def test_the_response_curve_refuses_to_extrapolate(fixture_model):
    model, _ = fixture_model
    curve = model.response_curve(["anxiety"], [0.0, 1.0, 2.0])
    assert curve("anxiety", 1.0) == pytest.approx(
        model.index("anxiety", model.rates(1.0)), abs=1e-9
    )
    with pytest.raises(ValueError):
        curve("anxiety", 2.5)
    with pytest.raises(ValueError):
        curve("anxiety", -0.1)


def test_the_curve_reproduces_the_integrator_at_its_own_grid_points(
        fixture_model):
    model, _ = fixture_model
    grid = [0.0, 0.5, 1.5, 3.0]
    curve = model.response_curve(["anxiety", "locomotion"], grid)
    for x in grid:
        rates = model.rates(x)
        for key in ("anxiety", "locomotion"):
            assert curve(key, x) == pytest.approx(
                model.index(key, rates), abs=1e-9
            )


def test_the_interpolation_error_is_measured_and_small(fixture_model):
    """The shortcut that makes the fit affordable, priced rather than
    assumed. The tolerance here is not a scientific tolerance: it is far
    below the smallest likelihood width any P7.3 target carries (0.05 on a
    ratio of order 1), so an interpolated prediction and an integrated one
    are the same datum as far as the posterior is concerned."""
    model, _ = fixture_model
    grid = np.concatenate([[0.0], np.geomspace(0.02, 5.0, 24)])
    curve = model.response_curve(["anxiety", "locomotion"], grid)
    error = curve.interpolation_error(model)
    for key, value in error.items():
        assert value < 1e-3, (key, value)


def test_a_response_curve_needs_an_increasing_grid():
    with pytest.raises(ValueError):
        efwd.ResponseCurve("x", [0.0, 1.0, 1.0], {"anxiety": [0.0, 1.0, 2.0]})


# ---------------------------------------------------------------------------
# The full 291-region build
# ---------------------------------------------------------------------------

_FULL_DATA = (
    fetch.is_cached("knox2019_connection_strength")
    and fetch.is_cached("knox2019_connection_density")
    and fetch.is_cached(sources.ONTOLOGY_KEY)
)

full_data_only = pytest.mark.skipif(
    not _FULL_DATA,
    reason="needs the gitignored data/connectome cache; "
    "run scripts/fetch_connectome.py",
)


@pytest.fixture(scope="module")
def whole_brain():
    if not _FULL_DATA:
        pytest.skip("no cache")
    return efwd.build_whole_brain_model(
        "Gabra1", "connection_modulation", name="test/291",
    )


@full_data_only
def test_no_endpoint_reads_a_region_the_atlas_does_not_measure(whole_brain):
    """The decision the 12 unmeasured regions force, checked rather than
    argued: excluding them from the drug's action touches no endpoint
    directly, so it acts only through the network."""
    model, provenance = whole_brain
    assert len(provenance["unmeasured_nodes"]) == 12
    for key in endpoints.ENDPOINTS:
        assert model.endpoint_regions_without_density(key) == (), key


@full_data_only
def test_the_whole_brain_operating_point_is_the_one_p3_4_certifies(
        whole_brain):
    model, provenance = whole_brain
    assert provenance["n_nodes"] == 291
    assert provenance["operating_gain"] == stability.DEFAULT_OPERATING_GAIN
    assessment = stability.assess(model.network)
    assert assessment.ok


@full_data_only
def test_every_endpoint_resolves_with_nothing_unmapped(whole_brain):
    model, _ = whole_brain
    for key in endpoints.ENDPOINTS:
        resolved = model.resolved_endpoint(key)
        assert resolved.unmapped == (), key
        assert resolved.readout.weights, key
