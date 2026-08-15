"""P8.2: the forward path from a binding fingerprint to a whole-brain endpoint.

Split the way this repository splits its calibration tests: the RULE (what the
composition must do, checked where the answer is known by construction), the
DERIVATION (that every standing this module reports is re-derived from the
module it claims to read, not restated), and the NUMBERS (measured facts about
the built model, asserted so they cannot drift silently).
"""

import numpy as np
import pytest

from neurocascade import fingerprint as fp
from neurocascade.calibration import endpoint_forward as efwd
from neurocascade.calibration import heldout_forward as hf


# ---------------------------------------------------------------------------
# The transduction sign and the gain standing, re-derived
# ---------------------------------------------------------------------------


RECEPTOR_TARGETS = tuple(
    sorted(name for name, spec in fp.TARGETS.items() if not spec.is_transporter)
)
TRANSPORTER_TARGETS = tuple(
    sorted(name for name, spec in fp.TARGETS.items() if spec.is_transporter)
)


@pytest.mark.parametrize("target", RECEPTOR_TARGETS)
def test_every_receptor_target_has_one_mechanism_and_one_transduction_sign(target):
    """No receptor may declare couplings of both signs, and none may declare
    both mechanisms. This is the whole basis on which a direction is predicted
    without a gain, so it is checked for all twenty, not for a sample."""
    mechanism, sign = hf.transduction_sign(target)
    assert mechanism in efwd.MECHANISMS
    assert sign in (-1, 1)


@pytest.mark.parametrize("target", TRANSPORTER_TARGETS)
def test_a_transporter_has_no_transduction_sign_and_says_why(target):
    with pytest.raises(ValueError, match="transporter"):
        hf.transduction_sign(target)


def test_the_transduction_signs_are_the_ones_the_p5_modules_declare():
    """Read the sign off each system's own coupling dict here too, by a
    different route than the module under test uses, so a change to a P5
    module's mechanism shows up as a failure rather than propagating."""
    for target in RECEPTOR_TARGETS:
        system = fp.TARGETS[target].system()
        declared = list(
            (dict(getattr(system, "coupling", None) or {})
             or dict(getattr(system, "connection_modulation", None) or {})).values()
        )
        expected = int(np.sign([v for v in declared if v != 0.0][0]))
        assert hf.transduction_sign(target)[1] == expected, target


def test_the_gi_coupled_receptors_are_the_ones_with_a_negative_sign():
    """A spot check against pharmacology rather than against the code, so
    that a sign table which is internally consistent and wrong still fails.
    Every receptor named here is canonically inhibitory (Gi/Go) or, for
    5HT1A, both Gi and a somatodendritic autoreceptor."""
    for target in ("D2", "5HT1A", "alpha2", "M2", "MOR", "DOR"):
        assert hf.transduction_sign(target)[1] == -1, target
    for target in ("D1", "5HT2A", "H1", "M1", "alpha1", "beta", "A2A",
                   "alpha4beta2", "alpha7"):
        assert hf.transduction_sign(target)[1] == +1, target


def test_the_modulated_edge_sign_splits_the_five_systems_three_two():
    """The correction this module makes to `endpoint_forward`'s hardcoded
    "inhibitory edges only". NMDA, AMPA and KOR scale excitatory edges;
    GABA-A's benzodiazepine site and CB1 scale inhibitory ones. Getting this
    wrong gives a whole-brain NMDA antagonist the sign of an agonist."""
    assert hf.modulated_edge_sign("NMDA") == +1
    assert hf.modulated_edge_sign("AMPA") == +1
    assert hf.modulated_edge_sign("KOR") == +1
    assert hf.modulated_edge_sign("GABA_A_BZD") == -1
    assert hf.modulated_edge_sign("CB1") == -1


def test_a_drive_coupled_target_has_no_edge_sign():
    with pytest.raises(ValueError, match="modulates no edge"):
        hf.modulated_edge_sign("D2")


@pytest.mark.parametrize("target,network", sorted(
    hf.MODULATED_EDGE_NETWORK.items()
))
def test_each_recorded_network_really_contains_that_systems_declared_edges(
    target, network
):
    """`MODULATED_EDGE_NETWORK` is a pointer to where the sign is read from,
    so the pointer has to resolve. A stale entry would otherwise raise a
    KeyError deep inside a report."""
    import importlib

    module_name, builder = network
    net = getattr(importlib.import_module("neurocascade.%s" % module_name),
                  builder)()
    system = fp.TARGETS[target].system()
    edges = dict(system.connection_modulation or {})
    assert edges, target
    for post, pre in edges:
        assert post in net.index, (target, post)
        assert pre in net.index, (target, pre)


def test_the_three_non_illustrative_gains_are_exactly_the_ones_flagged():
    """The claim that most gains are unfitted is the reason this task reports
    reachable ranges instead of magnitudes, so which ones are NOT unfitted is
    pinned exactly rather than merely being non-empty."""
    non_illustrative = {
        t for t in fp.TARGETS if hf.gain_standing(t)[0] != "unfitted"
    }
    assert non_illustrative == {"NMDA", "AMPA", "CB1", "KOR"}
    assert hf.gain_standing("NMDA")[0] == "pinned-by-algebra"
    assert hf.gain_standing("AMPA")[0] == "pinned-by-algebra"
    assert hf.gain_standing("CB1")[0] == "measured"
    assert hf.gain_standing("KOR")[0] == "measured"


def test_the_declared_gains_are_their_modules_own_constants():
    from neurocascade import cannabinoid, glutamate, opioid

    assert hf.declared_gain("NMDA") == glutamate.GLUTAMATE_CONNECTION_GAIN
    assert hf.declared_gain("AMPA") == glutamate.GLUTAMATE_CONNECTION_GAIN
    assert hf.declared_gain("CB1") == cannabinoid.CB1_CONNECTION_GAIN
    assert hf.declared_gain("KOR") == opioid.KOR_CONNECTION_GAIN


def test_asking_for_an_unfitted_gains_value_raises_rather_than_returning_one():
    """The one refusal in this module that costs a caller something. Handing
    back an illustrative number here is how an invented magnitude would get
    into a prediction."""
    with pytest.raises(ValueError, match="unfitted and illustrative"):
        hf.declared_gain("D2")


# ---------------------------------------------------------------------------
# The activation change, and what it is zero for
# ---------------------------------------------------------------------------


def test_an_antagonist_at_a_site_with_no_tone_changes_no_activation():
    """Not a modelling shortcut: displacing nothing changes nothing. This is
    why P6.2's measured tone is what makes an antagonist predictable, and why
    a compound acting only at tone-free sites gets no behavioural magnitude."""
    for target, antagonist in (("MOR", "naloxone"),
                               ("GABA_A_BZD", "flumazenil"),
                               ("A2A", "caffeine")):
        site = fp.TARGETS[target].system().receptor_site
        delta, baseline, _ = hf.activation_change(target, site, antagonist, 1e4)
        assert baseline == 0.0, target
        assert delta == 0.0, target


def test_an_antagonist_at_a_site_with_tone_removes_activation():
    """D2 is the case that matters: P6.2's measured striatal dopamine holds
    71% of the calibrated D2 pool, so a D2 blocker has something to displace
    and a whole-brain direction to predict."""
    site = fp.TARGETS["D2"].system().receptor_site
    delta, baseline, with_drug = hf.activation_change("D2", site, "haloperidol", 1e4)
    assert baseline == pytest.approx(0.71114, abs=1e-4)
    assert delta < 0.0
    assert with_drug < baseline


def test_an_agonist_at_a_tone_free_site_reproduces_its_own_activation():
    """With no tone the baseline leg is exactly zero, so delta_activation is
    the drug's own activation - which is what every scenario before P6.2
    computed, and is the sense in which this generalises rather than replaces
    them."""
    site = fp.TARGETS["MOR"].system().receptor_site
    delta, baseline, with_drug = hf.activation_change("MOR", site, "fentanyl", 100.0)
    assert baseline == 0.0
    assert delta == pytest.approx(with_drug)
    assert 0.0 < delta <= 1.0


def test_a_compound_absent_from_the_site_raises_rather_than_scoring_zero():
    site = fp.TARGETS["D2"].system().receptor_site
    with pytest.raises(ValueError, match="not a ligand"):
        hf.activation_change("D2", site, "morphine", 100.0)


def test_the_histamine_tone_bounds_what_an_h1_blocker_can_ever_do():
    """A measured fact with a consequence: [Itoh1991]'s 7.8 nM histamine sits
    ~300-fold below H1's Kd, so H1 antagonism can remove at most 0.32% of
    activation no matter the dose. Any compound whose predicted sedation runs
    through H1 is bounded by this number, not by its own affinity."""
    site = fp.TARGETS["H1"].system().receptor_site
    delta, baseline, _ = hf.activation_change("H1", site, "mepyramine", 1e6)
    assert baseline == pytest.approx(0.00324, abs=1e-5)
    assert abs(delta) <= baseline
    assert abs(delta) < 0.004


@pytest.mark.parametrize("target,status", [
    ("DAT", "transporter-no-whole-brain-field"),
    ("SERT", "transporter-no-whole-brain-field"),
    ("NET", "transporter-no-whole-brain-field"),
])
def test_a_transporter_has_no_behavioural_path_and_is_named_not_zeroed(
    target, status
):
    assert hf.behavioural_path_status(target, 0.0, set()) == status


def test_the_path_statuses_distinguish_no_tone_from_no_map():
    genes = {fp.TARGETS[t].gene for t in RECEPTOR_TARGETS}
    assert hf.behavioural_path_status("MOR", 0.0, genes) == "antagonist-no-tone"
    assert hf.behavioural_path_status("D2", 0.0, genes) == "no-activation-change"
    assert hf.behavioural_path_status("D2", -0.5, genes) == "available"
    assert hf.behavioural_path_status("D2", -0.5, set()) == "no-expression-map"


# ---------------------------------------------------------------------------
# The whole-brain composition
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def multi_model():
    model, provenance = hf.build_multi_target_model(
        ["MOR", "D2", "GABA_A_BZD", "NMDA", "DAT"], "test-multi"
    )
    return model, provenance


def test_a_transporter_is_excluded_by_name_rather_than_dropped(multi_model):
    _, provenance = multi_model
    assert provenance["excluded"] == {"DAT": "transporter-no-whole-brain-field"}


def test_one_layer_reduces_exactly_to_the_p7_3_single_target_model(multi_model):
    """The invariant that makes this a composition rather than a second
    implementation. Bit-for-bit, not to a tolerance: the two build the same
    coupling dict over the same network, so any difference at all is a bug
    rather than an integrator artefact."""
    model, _ = multi_model
    single, _ = efwd.build_whole_brain_model("Oprm1", "drive_coupling", "one")
    for x in (0.0, 2.0, -7.5):
        assert np.array_equal(model.rates({"MOR": x}), single.rates(x))


def test_the_inhibitory_edge_case_also_reduces_exactly(multi_model):
    model, _ = multi_model
    single, _ = efwd.build_whole_brain_model(
        "Gabra1", "connection_modulation", "one"
    )
    for x in (3.0, -3.0):
        assert np.array_equal(model.rates({"GABA_A_BZD": x}), single.rates(x))


def test_an_excitatory_modulator_does_not_match_the_inhibitory_edge_build(
    multi_model
):
    """The correction, measured. `endpoint_forward` would put an NMDA blocker
    on inhibitory edges; this puts it on excitatory ones, and the two disagree
    by far more than any numerical tolerance. If this ever passes at zero the
    edge-sign fix has been silently reverted."""
    model, _ = multi_model
    wrong, _ = efwd.build_whole_brain_model(
        "Grin1", "connection_modulation", "wrong"
    )
    difference = np.abs(model.rates({"NMDA": 3.0}) - wrong.rates(3.0)).max()
    assert difference > 0.1


def test_an_absent_target_is_the_same_as_zero(multi_model):
    model, _ = multi_model
    assert np.array_equal(model.rates({}), model.rates({"MOR": 0.0}))
    assert np.array_equal(model.rates({"MOR": 1.5}),
                          model.rates({"MOR": 1.5, "D2": 0.0}))


def test_the_drug_free_index_is_one_half_for_every_endpoint(multi_model):
    """P7.3's property, carried through the multi-target composition. It is
    what makes a drug/vehicle RATIO of the index well defined, and it holds
    because every endpoint's weights have unit L1 norm and P3.4 balances
    every node to 0.5."""
    model, _ = multi_model
    for key, value in model.indices({}).items():
        assert value == pytest.approx(efwd.BASELINE_INDEX, abs=1e-9), key


def test_two_targets_together_are_not_either_one_alone(multi_model):
    """A weak assertion on purpose. The composition is additive in the drive,
    not in the index, because the steady state is nonlinear; asserting an
    additive index would assert something false."""
    model, _ = multi_model
    joint = model.indices({"MOR": 2.0, "D2": 2.0}, ["locomotion"])["locomotion"]
    mor = model.indices({"MOR": 2.0}, ["locomotion"])["locomotion"]
    d2 = model.indices({"D2": 2.0}, ["locomotion"])["locomotion"]
    assert joint != pytest.approx(mor)
    assert joint != pytest.approx(d2)
    assert joint > max(mor, d2)


# ---------------------------------------------------------------------------
# The two gain-free quantities
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mor_response(multi_model):
    """One sweep, reused by every test below.

    A coarse grid on purpose: these tests check the RULES the two gain-free
    quantities obey, and those hold at any grid density. The report script
    uses the full `response_grid`, and its interpolation error against a
    re-integration is a measured number in the results file.
    """
    model, _ = multi_model
    grid = hf.response_grid(points_per_decade=2)
    return model, hf.sweep_target(model, "MOR", x_grid=grid)


def test_the_response_grid_is_symmetric_and_contains_zero():
    grid = hf.response_grid()
    assert 0.0 in grid
    assert np.allclose(grid, -grid[::-1])
    assert np.all(np.diff(grid) > 0)


def test_every_reachable_range_converged(mor_response):
    """The whole value of a reachable range is that it bounds the model at
    EVERY gain. A range still growing at the edge of the sweep would bound
    nothing, so convergence is checked rather than assumed."""
    _, response = mor_response
    for key, reachable in hf.reachable_index_range(response).items():
        assert reachable.converged, (key, reachable.drift)


def test_a_reachable_range_brackets_the_drug_free_baseline(mor_response):
    _, response = mor_response
    for key, reachable in hf.reachable_index_range(response).items():
        assert reachable.lo <= efwd.BASELINE_INDEX <= reachable.hi, key
        assert reachable.ratio_lo <= 1.0 <= reachable.ratio_hi, key


def test_analgesia_is_the_narrowest_reachable_endpoint(mor_response):
    """A measured structural fact with real consequences. The analgesia
    readout saturates over a much smaller span than locomotion, so a large
    published analgesic effect is outside the model's reach at ANY gain -
    a refutation that costs no parameter choice at all."""
    _, response = mor_response
    ranges = hf.reachable_index_range(response)
    widths = {k: r.ratio_hi - r.ratio_lo for k, r in ranges.items()}
    assert min(widths, key=widths.get) == "analgesia"
    assert widths["analgesia"] < 0.5
    assert widths["locomotion"] > 1.0


def test_shortfall_is_zero_inside_and_positive_outside(mor_response):
    _, response = mor_response
    reachable = hf.reachable_index_range(response)["analgesia"]
    assert reachable.shortfall(1.0) == 0.0
    assert reachable.contains_ratio(1.0)
    assert not reachable.contains_ratio(10.0)
    assert reachable.shortfall(10.0) == pytest.approx(10.0 - reachable.ratio_hi)


def test_implied_x_round_trips_through_the_tabulated_curve(mor_response):
    """The inversion is checked by evaluating the curve at the root, which is
    the only way to know it inverted this function rather than another."""
    _, response = mor_response
    for x_true in (0.8, -1.6):
        ratio = float(response("locomotion", x_true)) / response.baseline["locomotion"]
        x_found = hf.implied_x(response, "locomotion", ratio)
        assert x_found is not None
        recovered = float(response("locomotion", x_found)) / response.baseline["locomotion"]
        assert recovered == pytest.approx(ratio, abs=1e-9)


def test_implied_x_returns_none_outside_the_reachable_range(mor_response):
    """The two quantities have to agree: an observation the range calls
    unreachable must have no implied gain, or one of them is lying."""
    _, response = mor_response
    reachable = hf.reachable_index_range(response)["analgesia"]
    unreachable = reachable.ratio_hi + 0.5
    assert not reachable.contains_ratio(unreachable)
    assert hf.implied_x(response, "analgesia", unreachable) is None


def test_the_curve_refuses_to_extrapolate(mor_response):
    _, response = mor_response
    with pytest.raises(ValueError, match="outside the tabulated range"):
        response("locomotion", 10.0 * response.x_grid[-1])
