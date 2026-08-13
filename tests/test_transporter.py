"""P5.8: monoamine transporters DAT, SERT and NET.

Three groups of tests, in the order the module is built:

1. the MECHANISM (`transport.py`): that a transporter is structurally not a
   receptor, that a pure blocker cannot release transmitter, and that the
   steady-state law the module's headline numbers rest on is the one the
   integrated ODE actually produces;
2. the CONSTANTS (`monoamine_transporters.py`): that every number is the one
   its citation says, that the gaps are recorded as gaps, and that the
   cross-preparation and cross-species disagreements are asserted rather
   than hidden;
3. the ENGINE and the PHASE 6 INTERFACE: that a transporter system slots into
   the shared state vector without moving any pre-P5.8 result, and that a
   receptor driven from a transmitter pool behaves the way the pharmacology
   says.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from neurocascade import monoamine_transporters as mt
from neurocascade import scenarios
from neurocascade.engine import ReceptorSystem, Simulation, TransporterSystem
from neurocascade.pk import OneCompartmentPK
from neurocascade.receptor import ReceptorSite
from neurocascade.transport import MonoamineTransporter, TransporterLigand


# ---------------------------------------------------------------------------
# 1. The mechanism
# ---------------------------------------------------------------------------


def test_a_transporter_is_structurally_not_a_receptor():
    """The brief's own sentence, as a structural fact rather than a name.

    A `ReceptorSite` has `activation`, its ligands have `efficacy`, and its
    state is an occupancy. A transporter has none of the three: its output is
    a concentration, so there is nothing for a circuit node to read directly.
    """
    transporter = mt.dat()
    assert not hasattr(transporter, "activation")
    assert not hasattr(transporter, "occupancy_rhs")
    for ligand in transporter.ligands:
        assert not hasattr(ligand, "efficacy")
        assert not hasattr(ligand, "kon")
        assert not hasattr(ligand, "koff")
    # And the thing it does have is the transmitter concentration.
    assert transporter.n_states == 1
    assert transporter.initial_state().shape == (1,)


def test_transporter_system_has_no_coupling_of_any_kind():
    """A transporter contributes nothing to drive and nothing to
    connectivity. If either of these attributes ever appears, the mechanism
    has quietly become a receptor."""
    system = mt.transporter_system("DAT")
    assert not hasattr(system, "coupling")
    assert not hasattr(system, "connection_modulation")


def test_a_ligand_with_no_mechanism_is_rejected():
    with pytest.raises(ValueError, match="neither an inhibition constant"):
        TransporterLigand(name="nothing")


def test_a_substrate_with_no_releasing_efficacy_is_rejected():
    with pytest.raises(ValueError, match="release EC50 but no releasing"):
        TransporterLigand(name="odd", ki=100.0, release_ec50=10.0, release_emax=0.0)


def test_substrate_and_blocker_are_classified_by_the_measurement():
    """`is_substrate` is exactly "a release EC50 was measured", which is
    [Rothman2012]'s substrate-reversal criterion and not a drug-class label."""
    assert mt.SUBSTRATES == frozenset(["amphetamine"])
    assert mt.PURE_BLOCKERS == frozenset(
        ["cocaine", "methylphenidate", "fluoxetine", "atomoxetine"]
    )
    for transporter in mt.TRANSPORTERS:
        for ligand in mt.TRANSPORTER_BUILDERS[transporter]().ligands:
            assert ligand.is_substrate == mt.is_substrate(ligand.name)
            assert ligand.is_blocker == (
                not mt.is_substrate(ligand.name) and ligand.ki is not None
            )


def test_a_pure_blocker_cannot_release_transmitter_at_any_concentration():
    """The identity that makes the distinction structural: with only pure
    blockers on board the efflux term is exactly zero, at every
    concentration, for every transporter."""
    for transporter in mt.TRANSPORTERS:
        built = mt.TRANSPORTER_BUILDERS[transporter]()
        for scale in (1.0, 1e3, 1e6, 1e9):
            concentrations = np.array(
                [
                    0.0 if mt.is_substrate(lig.name) else scale
                    for lig in built.ligands
                ]
            )
            assert built.release_drive(concentrations) == 0.0
            assert built.efflux(concentrations) == 0.0


def test_a_substrate_also_competes_for_the_substrate_site():
    """Amphetamine is not "a releaser instead of a blocker": it is both, and
    the inhibition factor includes it."""
    dat = mt.dat()
    concentrations = np.array(
        [1000.0 if lig.name == "amphetamine" else 0.0 for lig in dat.ligands]
    )
    assert dat.inhibition_factor(concentrations) == pytest.approx(
        1.0 + 1000.0 / mt.ki("amphetamine", "DAT")
    )
    assert dat.release_drive(concentrations) > 0.0


#: min. Long enough for the SLOWEST pool here to settle. The relaxation time
#: constant of a transmitter pool is `Km * a / Vmax`, which is 52 ms for
#: dopamine and about half a minute for serotonin under
#: heavy blockade (Vmax at SERT is 39-fold below DAT's, [JohnJones2007]), so
#: a five-minute integration leaves serotonin 7e-5 short of its steady state.
#: That is a statement about the measured constants, not about tolerance.
_SETTLE_MINUTES = 120.0


def _integrate_to_steady_state(transporter, concentrations, t_end=_SETTLE_MINUTES):
    sol = solve_ivp(
        lambda t, y: transporter.transmitter_rhs(y, concentrations),
        (0.0, t_end),
        transporter.initial_state(),
        method="LSODA",
        rtol=1e-10,
        atol=1e-12,
    )
    assert sol.success
    return float(sol.y[0, -1])


def test_the_drug_free_steady_state_is_exactly_the_measured_basal_level():
    for build in (mt.dat, mt.sert, mt.net):
        transporter = build()
        zeros = np.zeros(transporter.n)
        settled = _integrate_to_steady_state(transporter, zeros)
        assert settled == pytest.approx(transporter.basal, rel=1e-9)


def test_blocker_steady_state_is_exactly_one_plus_C_over_Ki():
    """The law the module's headline table is built on, checked against the
    integrated ODE rather than against the algebra that produced it."""
    for transporter_name in mt.TRANSPORTERS:
        transporter = mt.TRANSPORTER_BUILDERS[transporter_name]()
        for drug in ("cocaine", "methylphenidate"):
            constant = mt.ki(drug, transporter_name)
            for concentration in (100.0, 1000.0, 10000.0):
                doses = np.array(
                    [
                        concentration if lig.name == drug else 0.0
                        for lig in transporter.ligands
                    ]
                )
                predicted = transporter.basal * (1.0 + concentration / constant)
                assert _integrate_to_steady_state(
                    transporter, doses
                ) == pytest.approx(predicted, rel=1e-6)
                assert mt.fold_rise(
                    drug, concentration, transporter_name
                ) == pytest.approx(1.0 + concentration / constant)


def test_the_blocker_fold_rise_does_not_depend_on_km_vmax_or_basal():
    """The cancellation stated in `transport.py`, made a test: change every
    illustrative-or-uncertain constant and the predicted fold-rise does not
    move. This is why the transporter layer's central prediction carries no
    illustrative parameter."""
    ligands = mt.ligands_at("DAT")
    reference = None
    for km, vmax, basal in [
        (mt.KM_DOPAMINE, mt.VMAX_DOPAMINE, mt.BASAL_DOPAMINE),
        (mt.KM_DOPAMINE, mt.VMAX_DOPAMINE_SECOND_LAB, mt.BASAL_DOPAMINE),
        (600.0, 1e5, 20.0),
        (17.0, 9e6, 0.4),
    ]:
        transporter = MonoamineTransporter(
            name="probe",
            transmitter="dopamine",
            ligands=[
                TransporterLigand(name=l.name, ki=l.ki) for l in ligands
            ],
            km=km,
            vmax=vmax,
            basal=basal,
        )
        doses = np.array(
            [1000.0 if l.name == "cocaine" else 0.0 for l in transporter.ligands]
        )
        fold = _integrate_to_steady_state(transporter, doses) / basal
        if reference is None:
            reference = fold
        assert fold == pytest.approx(reference, rel=1e-6)
    assert reference == pytest.approx(1.0 + 1000.0 / mt.ki("cocaine", "DAT"), rel=1e-6)


def test_the_first_order_form_is_the_linear_limit_of_the_michaelis_menten_one():
    """NET is built in the first-order form because only `Vmax/Km` is
    published for it. That is a limit of the same law, not a different one,
    and at the basal levels these constants imply the two agree to better
    than 3%."""
    ligands = [TransporterLigand(name="cocaine", ki=mt.ki("cocaine", "DAT"))]
    michaelis = MonoamineTransporter(
        name="mm",
        transmitter="dopamine",
        ligands=list(ligands),
        km=mt.KM_DOPAMINE,
        vmax=mt.VMAX_DOPAMINE,
        basal=mt.BASAL_DOPAMINE,
    )
    linear = MonoamineTransporter.first_order(
        name="linear",
        transmitter="dopamine",
        ligands=list(ligands),
        clearance_rate=mt.VMAX_DOPAMINE / mt.KM_DOPAMINE,
        basal=mt.BASAL_DOPAMINE,
    )
    assert michaelis.effective_clearance_rate == pytest.approx(
        linear.effective_clearance_rate
    )
    for concentration in (0.0, 100.0, 1000.0):
        doses = np.array([concentration])
        a = _integrate_to_steady_state(michaelis, doses)
        b = _integrate_to_steady_state(linear, doses)
        assert a == pytest.approx(b, rel=0.03)


def test_the_two_forms_are_mutually_exclusive_at_construction():
    with pytest.raises(ValueError, match="not both"):
        MonoamineTransporter(
            name="x",
            transmitter="dopamine",
            ligands=[TransporterLigand(name="c", ki=1.0)],
            km=1.0,
            vmax=1.0,
            clearance_rate=1.0,
            basal=1.0,
        )
    with pytest.raises(ValueError, match="needs both km and vmax"):
        MonoamineTransporter(
            name="x",
            transmitter="dopamine",
            ligands=[TransporterLigand(name="c", ki=1.0)],
            km=1.0,
            basal=1.0,
        )


def test_the_derived_basal_release_rate_is_what_the_mass_balance_requires():
    dat = mt.dat()
    assert dat.basal_release_rate() == pytest.approx(
        mt.VMAX_DOPAMINE * mt.BASAL_DOPAMINE / (mt.KM_DOPAMINE + mt.BASAL_DOPAMINE)
    )
    assert dat.uptake(dat.basal, np.zeros(dat.n)) == pytest.approx(
        dat.basal_release_rate()
    )
    net = mt.net()
    assert net.basal_release_rate() == pytest.approx(
        mt.NORADRENALINE_CLEARANCE_RATE * mt.BASAL_NORADRENALINE
    )


def test_release_rate_is_a_plain_attribute_because_phase_6_replaces_it():
    """The documented hand-off: P6.1 drives release from a source nucleus's
    firing rate, and the only thing it has to touch is this number."""
    dat = mt.dat()
    dat.release_rate = 2.0 * dat.basal_release_rate()
    settled = _integrate_to_steady_state(dat, np.zeros(dat.n))
    assert settled > 1.9 * mt.BASAL_DOPAMINE


# ---------------------------------------------------------------------------
# 2. The constants
# ---------------------------------------------------------------------------


def test_all_five_drugs_the_brief_names_are_present_with_constants():
    assert set(mt.DRUGS) == {
        "cocaine",
        "methylphenidate",
        "amphetamine",
        "fluoxetine",
        "atomoxetine",
    }
    for drug in mt.DRUGS:
        cells = [t for t in mt.TRANSPORTERS if mt.ki(drug, t) is not None]
        assert cells, "no constant at all for %s" % drug


def test_the_missing_cells_are_recorded_rather_than_filled_in():
    """Fluoxetine at NET and at DAT have no obtainable constant. The model
    must show that as an absence, not as an implied infinite Ki."""
    assert set(mt.MISSING_KI) == {("fluoxetine", "NET"), ("fluoxetine", "DAT")}
    for (drug, transporter) in mt.MISSING_KI:
        assert mt.ki(drug, transporter) is None
        assert drug not in mt.drugs_at(transporter)
        names = [lig.name for lig in mt.ligands_at(transporter)]
        assert drug not in names
    assert "fluoxetine" in mt.drugs_at("SERT")
    # And the bounded one says so, with its bound.
    assert "270" in mt.MISSING_KI[("fluoxetine", "DAT")]
    assert "Unbounded" in mt.MISSING_KI[("fluoxetine", "NET")]


def test_selectivity_refuses_to_mix_a_measured_cell_with_a_missing_one():
    assert mt.selectivity("fluoxetine", "SERT", "NET") is None
    assert mt.selectivity("atomoxetine", "NET", "SERT") == pytest.approx(77.0 / 5.0)
    assert mt.selectivity("methylphenidate", "DAT", "SERT") == pytest.approx(
        114370.0 / 260.0
    )


def test_release_is_far_more_potent_than_block_which_is_the_whole_point():
    """The number the substrate/blocker distinction is worth. Modelling
    amphetamine as a blocker alone would put its dopaminergic potency
    87.5-fold too low."""
    ratios = mt.release_over_block_ratio()
    assert set(ratios) == {"DAT", "SERT", "NET"}
    assert ratios["DAT"] == pytest.approx(560.0 / 6.4)
    assert ratios["DAT"] > 80.0
    assert ratios["NET"] == pytest.approx(120.0 / 7.4)
    assert ratios["SERT"] == pytest.approx(23820.0 / 1960.0)
    assert min(ratios.values()) > 10.0


def test_the_two_mouse_preparations_agree_at_DAT_and_disagree_at_SERT():
    """[HanGu2006]'s cloned mouse transporters against [JohnJones2007]'s
    mouse brain slices: both halves asserted, including the failures."""
    agreement = mt.preparation_agreement()
    for drug in ("cocaine", "methylphenidate", "amphetamine"):
        assert agreement[(drug, "DAT")] < 1.45, drug
    assert agreement[("cocaine", "SERT")] == pytest.approx(10000.0 / 730.0, rel=1e-6)
    assert agreement[("cocaine", "SERT")] > 13.0
    assert agreement[("methylphenidate", "SERT")] == pytest.approx(
        114370.0 / 19000.0, rel=1e-6
    )
    assert agreement[("amphetamine", "SERT")] < 1.1
    # And the one that is not a disagreement so much as a different world.
    assert agreement[("fluoxetine", "SERT")] > 1e4


def test_the_human_for_mouse_substitution_is_bounded_by_the_same_paper():
    """Fluoxetine and atomoxetine use human constants. [HanGu2006] measured
    nine cells in both species, so the price of that substitution is a
    measurement rather than a hope."""
    ratios = mt.species_ratio()
    assert len(ratios) == 9
    assert all(value >= 1.0 for value in ratios.values())
    worst = max(ratios, key=ratios.get)
    assert worst == ("methylphenidate", "DAT")
    assert ratios[worst] == pytest.approx(260.0 / 60.0)
    others = [v for k, v in ratios.items() if k != worst]
    assert max(others) < 2.5
    # The ordering that makes the mixing defensible: species costs less here
    # than the between-laboratory spread on atomoxetine's SERT cell.
    assert max(ratios.values()) < mt.atomoxetine_lab_disagreement()["SERT"]


def test_the_two_atomoxetine_laboratories_disagree_most_where_it_matters():
    spread = mt.atomoxetine_lab_disagreement()
    assert spread["DAT"] == pytest.approx(1451.0 / 1080.0)
    assert spread["NET"] == pytest.approx(5.0 / 2.03)
    assert spread["SERT"] == pytest.approx(77.0 / 8.9)
    # The consequence: NET-over-SERT selectivity is 15.4x on one paper's
    # numbers and 4.4x on the other's.
    used = mt.selectivity("atomoxetine", "NET", "SERT")
    other = mt.KI_SECOND_LAB["atomoxetine"]["SERT"] / mt.KI_SECOND_LAB["atomoxetine"]["NET"]
    assert used == pytest.approx(15.4)
    assert other == pytest.approx(4.384, rel=1e-3)


def test_fluoxetine_over_prediction_is_a_number_and_not_an_apology():
    """With a 3 nM SERT constant and no autoreceptor feedback the model says
    1 uM fluoxetine raises extracellular 5-HT 334-fold. Real SSRIs raise it a
    few fold. The gap is P6.1/P6.2's, and it is recorded as a number so it
    cannot be forgotten."""
    assert mt.fluoxetine_fold_rise(1000.0) == pytest.approx(334.333, rel=1e-4)
    assert mt.fluoxetine_fold_rise(1000.0) > 100.0


def test_fold_rise_refuses_to_be_used_for_a_substrate():
    with pytest.raises(ValueError, match="is a substrate"):
        mt.fold_rise("amphetamine", 100.0, "DAT")


def test_the_noradrenaline_clearance_rate_is_derived_from_the_measured_half_life():
    assert mt.NORADRENALINE_CLEARANCE_HALF_LIFE_S == 1.36
    assert mt.NORADRENALINE_CLEARANCE_RATE == pytest.approx(
        np.log(2.0) / (1.36 / 60.0)
    )
    assert mt.NORADRENALINE_CLEARANCE_RATE == pytest.approx(30.58, rel=1e-3)
    # NET has no Km and no Vmax, and that is the honest state of the
    # literature rather than an oversight.
    net = mt.net()
    assert not net.is_michaelis_menten
    assert net.km is None and net.vmax is None


def test_no_NET_conclusion_depends_on_the_noradrenaline_basal_level():
    """Every fold-change this module reports at NET must be invariant to the
    basal noradrenaline level.

    P5.8 wrote this test because `NORADRENALINE_BASAL_UNIT_SCALE` was a unit
    rather than a measurement, and the invariance was what made the unit
    harmless. P6.2 replaced the unit with [PanLai1995]'s measured 1.59 nM,
    and the test is KEPT and strengthened rather than retired: what it now
    asserts is that P6.2's change of that constant moved no fold-change this
    module reports, which is the claim `endogenous_tone.change_report()`
    makes for the transporter layer. The sweep still spans four orders of
    magnitude and now includes P6.1's own 1.0."""
    ligands = mt.ligands_at("NET")
    folds = []
    for basal in (mt.BASAL_NORADRENALINE, 1.0, 0.017, 42.0):
        transporter = MonoamineTransporter.first_order(
            name="NET",
            transmitter="noradrenaline",
            ligands=[
                TransporterLigand(
                    name=l.name,
                    ki=l.ki,
                    release_ec50=l.release_ec50,
                    release_emax=l.release_emax,
                )
                for l in ligands
            ],
            clearance_rate=mt.NORADRENALINE_CLEARANCE_RATE,
            basal=basal,
            release_capacity=mt.RELEASE_CAPACITY,
        )
        doses = np.array(
            [
                1000.0 if l.name in ("cocaine", "amphetamine") else 0.0
                for l in transporter.ligands
            ]
        )
        folds.append(_integrate_to_steady_state(transporter, doses) / basal)
    for other in folds[1:]:
        assert folds[0] == pytest.approx(other, rel=1e-9)


def test_the_substrate_conclusion_holds_for_any_positive_release_capacity():
    """`RELEASE_CAPACITY` is the module's one illustrative mechanism
    constant. The conclusion it could contaminate -- that amphetamine's
    releasing action dominates its blocking action -- must not depend on its
    value."""
    concentration = 100.0
    blockade_only = 1.0 + concentration / mt.ki("amphetamine", "DAT")
    for capacity in (0.05, 0.5, 1.0, 5.0, 50.0):
        dat = mt.dat(release_capacity=capacity)
        doses = np.array(
            [
                concentration if l.name == "amphetamine" else 0.0
                for l in dat.ligands
            ]
        )
        fold = _integrate_to_steady_state(dat, doses) / dat.basal
        assert fold > blockade_only
    # At zero capacity -- i.e. amphetamine modelled as a blocker, which is
    # what the brief says must not be done -- the release contribution is
    # gone entirely and the two coincide.
    dat = mt.dat(release_capacity=0.0)
    doses = np.array(
        [concentration if l.name == "amphetamine" else 0.0 for l in dat.ligands]
    )
    assert _integrate_to_steady_state(dat, doses) / dat.basal == pytest.approx(
        blockade_only, rel=1e-6
    )


def test_the_tool_compound_clearance_constant_is_shared_with_every_P5_module():
    from neurocascade import (
        acetylcholine,
        adenosine,
        dopamine,
        histamine,
        noradrenaline,
        opioid,
        serotonin,
    )

    for module in (
        serotonin,
        dopamine,
        noradrenaline,
        acetylcholine,
        histamine,
        adenosine,
        opioid,
    ):
        assert mt.TOOL_COMPOUND_KE == module.TOOL_COMPOUND_KE


def test_the_illustrative_constants_are_labelled_as_such():
    """Rule 1's bookkeeping: the numbers in this module that are not
    measurements say so where a reader will see them.

    P5.8 listed three. P6.2 removed `NORADRENALINE_BASAL_UNIT_SCALE` from the
    list in the same commit that supplied [PanLai1995]'s measurement of it
    (rule 3), and `test_endogenous_tone` asserts that the name is gone rather
    than merely relabelled."""
    assert "ILLUSTRATIVE" in mt.__doc__ or "illustrative" in mt.__doc__
    source = open(mt.__file__).read()
    for constant in (
        "RELEASE_CAPACITY",
        "TOOL_COMPOUND_KE",
    ):
        block = source.split("%s = " % constant)[0]
        # the doc-comment block immediately preceding the assignment
        preceding = block.rsplit("\n\n", 1)[-1]
        assert "ILLUSTRATIVE" in preceding, constant


# ---------------------------------------------------------------------------
# 3. The engine, and the Phase 6 interface
# ---------------------------------------------------------------------------


def test_a_simulation_with_no_transporters_is_unchanged_by_P5_8():
    """The regression that matters: every result predating P5.8 is produced
    by a Simulation with no transporter systems, whose state vector must be
    the same length, in the same order, with the same values."""
    sim = Simulation(
        network=scenarios.reward_circuit_network(),
        pk_models=scenarios.mu_opioid_pk_models(),
        receptor_site=scenarios.mu_opioid_receptor_site(),
        coupling=scenarios.mu_opioid_coupling(),
    )
    assert sim.m == 0
    assert sim.transporter_systems == []
    result = sim.run(dose_concentrations=[100.0, 0.0], t_span=(0.0, 10.0))
    assert result["transmitter"].shape == (0, result["t"].size)
    assert result["rates"].shape == (sim.c, result["t"].size)
    # The rates region still starts exactly where it used to, at d + r, so
    # no state moved: the transmitter block is empty and contributes no
    # offset.
    assert sim.d + sim.r + sim.m + sim.c == sim.d + sim.r + sim.c
    y0 = np.concatenate(
        [sim.initial_pk_state([100.0, 0.0]), np.zeros(sim.r), np.zeros(sim.c)]
    )
    assert sim.rhs(0.0, y0).shape == y0.shape


def test_a_transporter_only_simulation_is_legitimate():
    network = mt.stabilized_indirect_pathway_network()
    sim = Simulation(
        network=network, transporter_systems=mt.monoamine_transporter_systems()
    )
    assert sim.receptor_systems == []
    assert sim.r == 0
    assert sim.m == 3
    result = sim.run(
        doses=sim.dose_vector(cocaine=1000.0),
        t_span=(0.0, 5.0),
        rates0=np.full(network.n, 0.5),
        method="LSODA",
    )
    assert result["occupancy"].shape == (0, result["t"].size)
    assert result["transmitter"].shape == (3, result["t"].size)


def test_a_simulation_with_neither_kind_of_system_is_an_error():
    with pytest.raises(TypeError, match="transporter_systems"):
        Simulation(network=scenarios.reward_circuit_network())


def test_transporter_system_names_must_be_unique():
    with pytest.raises(ValueError, match="unique"):
        Simulation(
            network=mt.stabilized_indirect_pathway_network(),
            transporter_systems=[
                mt.transporter_system("DAT"),
                mt.transporter_system("DAT"),
            ],
        )


def test_the_transmitter_state_starts_at_the_measured_basal_level():
    sim = Simulation(
        network=mt.stabilized_indirect_pathway_network(),
        transporter_systems=mt.monoamine_transporter_systems(),
    )
    assert np.allclose(
        sim.initial_transmitter_state(),
        [mt.BASAL_DOPAMINE, mt.BASAL_SEROTONIN, mt.BASAL_NORADRENALINE],
    )


def test_a_drug_free_transporter_run_is_exactly_flat():
    """The one region of the state vector that does not start at zero has to
    start at its own fixed point, or every dosed-versus-baseline comparison
    in this module is measuring a start-up transient."""
    network, sim, result = mt.run(t_end=20.0)
    for name in mt.TRANSPORTERS:
        trace = mt.transmitter_trace(sim, result, name)
        assert np.allclose(trace, trace[0], rtol=1e-8)


def test_doses_are_ordered_receptor_systems_first():
    """Adding a transporter system must not renumber an existing receptor
    system's doses."""
    d2 = mt.d2_receptor_system_on_transporter_dopamine()
    sim = Simulation(
        network=mt.tonic_compensated_network(),
        receptor_systems=[d2],
        transporter_systems=mt.monoamine_transporter_systems(),
    )
    doses = sim.dose_vector(haloperidol=7.0, cocaine=3.0)
    assert doses[:2].tolist() == [0.0, 7.0]  # dopamine, haloperidol
    assert doses[2] == 3.0  # DAT's cocaine, first transporter ligand
    # Cocaine is at all three transporters, so one named dose sets three
    # entries -- the same "one drug, one dose, several sites" behaviour the
    # multi-receptor systems already have (see `opioid.pk_models`).
    assert (doses == 3.0).sum() == 3
    assert doses.sum() == 7.0 + 3.0 * 3


def test_transmitter_sources_must_name_a_ligand_and_a_transporter_present():
    site = scenarios.d2_receptor_site_calibrated()
    with pytest.raises(ValueError, match="not on this receptor site"):
        ReceptorSystem(
            pk_models=scenarios.d2_pk_models_calibrated(),
            receptor_site=site,
            coupling=scenarios.d2_coupling(),
            transmitter_sources={"serotonin": "SERT"},
        )
    system = ReceptorSystem(
        pk_models=scenarios.d2_pk_models_calibrated(),
        receptor_site=site,
        coupling=scenarios.d2_coupling(),
        transmitter_sources={"dopamine": "NOT_A_TRANSPORTER"},
    )
    with pytest.raises(ValueError, match="not in this Simulation"):
        Simulation(
            network=mt.stabilized_indirect_pathway_network(),
            receptor_systems=[system],
            transporter_systems=[mt.transporter_system("DAT")],
        )


def test_a_blocker_raises_all_three_transmitters_by_the_predicted_factors():
    """Cocaine is not a dopamine drug: its constants at the three mouse
    transporters span 1.6-fold, so it raises all three pools together."""
    network, sim, result = mt.run(cocaine=1000.0, t_end=5.0)
    for name in mt.TRANSPORTERS:
        fold = mt.transmitter_fold(sim, result, name)
        predicted = mt.fold_rise("cocaine", 1000.0, name)
        assert fold.max() == pytest.approx(predicted, rel=0.02)
    assert mt.fold_rise("cocaine", 1000.0, "NET") > mt.fold_rise(
        "cocaine", 1000.0, "SERT"
    )


def test_the_substrate_beats_the_blocker_at_a_tenth_of_the_concentration():
    """The headline consequence of representing the substrate mechanism.
    Amphetamine at 100 nM raises extracellular dopamine further than its own
    uptake blockade could, and comparably to cocaine at ten times the
    concentration."""
    _, sim_a, res_a = mt.run(amphetamine=100.0, t_end=5.0)
    amphetamine_fold = mt.transmitter_fold(sim_a, res_a, "DAT").max()
    blockade_only = mt.KI["amphetamine"]["DAT"]
    assert amphetamine_fold > 1.0 + 100.0 / blockade_only + 1.0

    _, sim_c, res_c = mt.run(cocaine=1000.0, t_end=5.0)
    cocaine_fold = mt.transmitter_fold(sim_c, res_c, "DAT").max()
    assert amphetamine_fold > 0.7 * cocaine_fold


def test_lsoda_and_rk45_agree_so_the_integrator_choice_is_not_the_model():
    """The measured uptake constants put more than three orders of magnitude
    of timescale separation into one state vector (P5.5's problem again). LSODA
    is the default for speed; the answer must not depend on it."""
    _, sim_a, res_a = mt.run(cocaine=1000.0, t_end=2.0, n_points=21, method="LSODA")
    _, sim_b, res_b = mt.run(cocaine=1000.0, t_end=2.0, n_points=21, method="RK45")
    # The real numbers, reported rather than rounded to a convenient
    # tolerance: the transmitter traces agree to 0.29% and the circuit rates
    # agree exactly, because nothing in this configuration couples the
    # transporter to the circuit. The 0.29% is RK45 under-resolving the
    # sub-millisecond transient at the first output point under
    # `max_step=0.1` min; it decays over the run.
    transmitter_error = np.abs(
        (res_a["transmitter"] - res_b["transmitter"]) / res_a["transmitter"]
    ).max()
    assert transmitter_error < 3e-3
    assert np.abs(res_a["rates"] - res_b["rates"]).max() < 1e-9


# --- the Phase 6 interface -------------------------------------------------


def test_the_resting_D2_occupancy_by_endogenous_dopamine_is_large():
    """Two measurements, no fitting: [Agren2021]'s dopamine Kd of 2.03 nM
    against [Bosse2011]'s basal 5.0 nM."""
    dopamine = scenarios.d2_dopamine_ligands_calibrated()[0]
    kd = dopamine.koff / dopamine.kon
    assert kd == pytest.approx(2.031, rel=1e-3)
    assert mt.resting_d2_occupancy() == pytest.approx(5.0 / (5.0 + kd))
    assert mt.resting_d2_occupancy() > 0.7


def test_without_tonic_compensation_the_target_node_saturates():
    """P3.4's lesson for the third time, asserted rather than asserted-away:
    the stabilization certificate is issued for the DRUG-FREE network, and
    tonic transmitter is not drug-free."""
    network = mt.stabilized_indirect_pathway_network()
    _, sim, result = mt.run(
        receptor_systems=[mt.d2_receptor_system_on_transporter_dopamine()],
        network=network,
        t_end=30.0,
    )
    striatum = result["rates"][network.index["Striatum_D2_MSN"], -1]
    assert striatum < 1e-3  # saturated OFF

    compensated = mt.tonic_compensated_network()
    _, sim2, result2 = mt.run(
        receptor_systems=[mt.d2_receptor_system_on_transporter_dopamine()],
        network=compensated,
        t_end=30.0,
    )
    assert np.allclose(result2["rates"][:, -1], 0.5, atol=1e-5)


def test_the_transporter_pool_drives_receptor_occupancy():
    """The interface, end to end: block DAT, extracellular dopamine rises,
    and the occupancy of a receptor that was given no dose of dopamine rises
    with it."""
    network = mt.tonic_compensated_network()
    _, sim0, base = mt.run(
        receptor_systems=[mt.d2_receptor_system_on_transporter_dopamine()],
        network=network,
        t_end=30.0,
    )
    _, sim1, dosed = mt.run(
        receptor_systems=[mt.d2_receptor_system_on_transporter_dopamine()],
        network=network,
        t_end=30.0,
        cocaine=1000.0,
    )
    assert base["occupancy"][0, -1] == pytest.approx(mt.resting_d2_occupancy(), rel=1e-4)
    assert dosed["transmitter"][sim1.transmitter_index["DAT"]].max() > 3.0 * mt.BASAL_DOPAMINE * 0.9
    assert dosed["occupancy"][0].max() > base["occupancy"][0].max()
    assert dosed["occupancy"][0].max() > 0.85
    # No dose of dopamine was given anywhere.
    assert sim1.dose_vector(cocaine=1000.0)[0] == 0.0


def test_a_psychostimulant_and_an_antipsychotic_move_the_readout_oppositely():
    """The circuit-level result, with both signs forced by the pharmacology
    rather than chosen: raising extracellular dopamine drives the Gi-coupled
    D2 receptor on the indirect-pathway MSN harder, which disinhibits the
    thalamus; blocking that receptor does the opposite."""
    network = mt.tonic_compensated_network()
    readout = scenarios.locomotor_activity_readout()
    scores = {}
    for label, doses in [
        ("baseline", {}),
        ("cocaine", {"cocaine": 1000.0}),
        ("amphetamine", {"amphetamine": 100.0}),
        ("haloperidol", {"haloperidol": 100.0}),
    ]:
        _, sim, result = mt.run(
            receptor_systems=[mt.d2_receptor_system_on_transporter_dopamine()],
            network=network,
            t_end=30.0,
            **doses
        )
        scores[label] = readout.score(network, result["rates"][:, -1])
    assert scores["baseline"] == pytest.approx(0.5, abs=1e-4)
    assert scores["cocaine"] > scores["baseline"] + 0.05
    assert scores["amphetamine"] > scores["baseline"] + 0.05
    assert scores["haloperidol"] < scores["baseline"] - 0.05


def test_an_antagonist_alone_is_no_longer_inert():
    """Every module from P5.1 to P5.7 had to write "an antagonist given alone
    is inert here, because there is no endogenous transmitter to displace".
    With a transporter maintaining a measured 5.0 nM dopamine pool, that
    sentence is false for the first time in this codebase."""
    network = mt.tonic_compensated_network()
    d2_with_tone = mt.d2_receptor_system_on_transporter_dopamine()
    _, _, base = mt.run(
        receptor_systems=[d2_with_tone], network=network, t_end=30.0
    )
    _, _, dosed = mt.run(
        receptor_systems=[mt.d2_receptor_system_on_transporter_dopamine()],
        network=network,
        t_end=30.0,
        haloperidol=100.0,
    )
    moved = np.abs(dosed["rates"][:, -1] - base["rates"][:, -1])
    assert moved.max() > 0.1
    assert dosed["occupancy"][0, -1] < 0.1  # dopamine displaced from the site
    assert dosed["occupancy"][1, -1] > 0.9  # by haloperidol

    # The control: the SAME receptor system with no transporter behind it,
    # which is what every pre-P5.8 configuration is, does nothing at all.
    plain = ReceptorSystem(
        pk_models=scenarios.d2_pk_models_calibrated(),
        receptor_site=scenarios.d2_receptor_site_calibrated(),
        coupling=scenarios.d2_coupling(),
    )
    sim = Simulation(network=mt.stabilized_indirect_pathway_network(),
                     receptor_systems=[plain])
    rates0 = np.full(network.n, 0.5)
    without = sim.run(doses=sim.dose_vector(), t_span=(0.0, 30.0), rates0=rates0,
                      method="LSODA")
    with_drug = sim.run(doses=sim.dose_vector(haloperidol=100.0),
                        t_span=(0.0, 30.0), rates0=rates0, method="LSODA")
    assert np.allclose(without["rates"][:, -1], with_drug["rates"][:, -1], atol=1e-9)


def test_the_transmitter_source_map_covers_every_transporter():
    assert set(mt.TRANSMITTER_SOURCE.values()) == set(mt.TRANSPORTERS)
    for transmitter, transporter in mt.TRANSMITTER_SOURCE.items():
        assert mt.TRANSMITTER[transporter] == transmitter


def test_a_transporter_ligand_carrying_no_ki_still_needs_a_mechanism():
    """`ki=None` is a recorded gap, not a silent zero-effect ligand: it is
    only constructible alongside a release EC50."""
    lig = TransporterLigand(name="x", release_ec50=10.0, release_emax=1.0)
    assert lig.ki is None
    assert lig.inhibition_term(1e6) == 0.0
    assert lig.is_substrate
    assert not lig.is_blocker


def test_receptor_sites_are_untouched_by_the_transporter_layer():
    """P5.8 adds a mechanism; it must not have changed the receptor one.
    A plain ReceptorSite still behaves exactly as before."""
    site = ReceptorSite(name="probe", ligands=scenarios.mu_opioid_ligands())
    occ = np.zeros(site.n)
    assert site.activation(occ) == 0.0
    assert site.n_states == site.n
    system = ReceptorSystem(
        pk_models=[OneCompartmentPK(ke=0.1) for _ in site.ligands],
        receptor_site=site,
        coupling={},
    )
    assert system.transmitter_sources == {}
