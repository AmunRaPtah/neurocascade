"""P1.5: the plasma -> brain partition layer, and the drug numbers in it.

Three groups of tests, in this order:

1. `brain_partition.BrainPartition` itself - the arithmetic, and above all
   the *refusals*. The refusals get more tests than the arithmetic because
   the arithmetic is one multiplication and the refusals are the whole point:
   a total-brain Kp and an unbound partition coefficient Kp,uu differ by
   fu_brain/fu_plasma, which is a factor of 10-150 for the drugs this project
   models, and nothing in a type system stops one being used as the other.
2. The `pk.py` / `engine.py` wiring - that occupancy is driven by the
   receptor-site concentration, and that a model with no partition is
   bit-for-bit what it was before P1.5.
3. The per-drug numbers in `drug_pk.py` - the derivations, the
   cross-checks against independent sources, and the guards that stop an
   unsourced partition being added later.
"""

import numpy as np
import pytest

from neurocascade import drug_pk, scenarios
from neurocascade.brain_partition import BrainPartition, UnusablePartitionError
from neurocascade.engine import ReceptorSystem, Simulation
from neurocascade.pk import OneCompartmentPK
from neurocascade.receptor import Ligand, ReceptorSite

BODY_WEIGHT_KG = 0.25


# ---------------------------------------------------------------------------
# 1. BrainPartition: arithmetic and refusals
# ---------------------------------------------------------------------------


def test_basis_must_be_stated_and_must_be_one_of_the_two():
    with pytest.raises(ValueError) as excinfo:
        BrainPartition(
            drug="x", species="rat", basis="brain:plasma", ratio=2.0, source="s"
        )
    assert "not interchangeable" in str(excinfo.value)


def test_a_source_is_required():
    with pytest.raises(ValueError) as excinfo:
        BrainPartition(drug="x", species="rat", basis="Kp", ratio=2.0, source="")
    assert "invented number" in str(excinfo.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ratio": 0.0},
        {"ratio": -1.0},
        {"fu_plasma": 0.0},
        {"fu_plasma": 1.5},
        {"fu_brain": -0.1},
    ],
)
def test_out_of_range_inputs_raise(kwargs):
    base = dict(drug="x", species="rat", basis="Kp", ratio=2.0, source="s")
    base.update(kwargs)
    with pytest.raises(ValueError):
        BrainPartition(**base)


def test_kp_only_cannot_produce_an_unbound_brain_concentration():
    """The single most important refusal in this module.

    A brain:plasma ratio of 22 says the tissue holds 22x what the plasma
    does; it says nothing about how much of that is free. Substituting it for
    a Kp,uu would drive the occupancy ODE with a concentration overstated by
    1/fu_brain.
    """
    p = BrainPartition(drug="x", species="rat", basis="Kp", ratio=22.0, source="s")
    assert p.total_brain_factor == 22.0
    assert p.kp == 22.0
    assert p.kp_uu is None
    assert p.can_drive_receptor is False
    with pytest.raises(UnusablePartitionError) as excinfo:
        p.unbound_brain_factor
    message = str(excinfo.value)
    assert "TOTAL brain" in message
    assert "fu_brain" in message


def test_kp_uu_without_a_plasma_free_fraction_is_also_refused():
    """The mirror-image mistake: Kp,uu relates *unbound* brain to *unbound*
    plasma, but `pk.py` produces *total* plasma, so an fu_plasma is needed to
    close the gap and is not defaulted to 1."""
    p = BrainPartition(drug="x", species="rat", basis="Kp,uu", ratio=1.0, source="s")
    assert p.can_drive_receptor is False
    with pytest.raises(UnusablePartitionError) as excinfo:
        p.unbound_brain_factor
    assert "fu_plasma" in str(excinfo.value)
    # ... and it cannot be turned back into a total-brain Kp either.
    with pytest.raises(UnusablePartitionError):
        p.total_brain_factor


def test_kp_uu_basis_with_fu_plasma_gives_the_right_factor():
    p = BrainPartition(
        drug="x", species="rat", basis="Kp,uu", ratio=0.8, fu_plasma=0.2, source="s"
    )
    assert p.kp_uu == 0.8
    assert p.unbound_brain_factor == pytest.approx(0.8 * 0.2)
    assert p.unbound_brain_concentration(50.0) == pytest.approx(8.0)


def test_kp_basis_with_fu_brain_gives_the_right_factor_and_the_identity():
    """Cu_brain = fu_brain * Kp * C_plasma, and Kp,uu = Kp * fu_b / fu_p.

    Note deliberately that the *driving* factor does not contain fu_plasma at
    all: fu_plasma is needed only to report Kp,uu. That asymmetry is why the
    diazepam entry's disagreement between two labs' fu_plasma values does not
    move the concentration this model drives the receptor with.
    """
    p = BrainPartition(
        drug="x",
        species="rat",
        basis="Kp",
        ratio=4.0,
        fu_plasma=0.2,
        fu_brain=0.05,
        source="s",
    )
    assert p.unbound_brain_factor == pytest.approx(4.0 * 0.05)
    assert p.kp_uu == pytest.approx(4.0 * 0.05 / 0.2)
    assert p.total_brain_factor == 4.0
    assert p.can_drive_receptor is True
    # Round trip: reconstructing on the Kp,uu basis gives the same driver.
    q = BrainPartition(
        drug="x",
        species="rat",
        basis="Kp,uu",
        ratio=p.kp_uu,
        fu_plasma=0.2,
        source="s",
    )
    assert q.unbound_brain_factor == pytest.approx(p.unbound_brain_factor)
    # ... but it can only report the *other* coefficient if both unbound
    # fractions are present, in either direction. Asymmetry by design: the
    # conversion Kp <-> Kp,uu needs fu_brain/fu_plasma whichever way it runs.
    assert q.kp is None
    q_full = BrainPartition(
        drug="x",
        species="rat",
        basis="Kp,uu",
        ratio=p.kp_uu,
        fu_plasma=0.2,
        fu_brain=0.05,
        source="s",
    )
    assert q_full.kp == pytest.approx(p.kp)


def test_species_mismatch_is_refused():
    p = BrainPartition(drug="x", species="rat", basis="Kp,uu", ratio=1.0, source="s")
    p.check_species("rat")
    p.check_species(None)  # a model that does not know its species opts out
    with pytest.raises(ValueError) as excinfo:
        p.check_species("mouse")
    assert "cross-species" in str(excinfo.value)


def test_equilibration_evidence_is_tracked_rather_than_assumed():
    silent = BrainPartition(
        drug="x", species="rat", basis="Kp,uu", ratio=1.0, fu_plasma=0.2, source="s"
    )
    assert silent.assumes_rapid_equilibrium_unverified is True
    evidenced = BrainPartition(
        drug="x",
        species="rat",
        basis="Kp,uu",
        ratio=1.0,
        fu_plasma=0.2,
        source="s",
        equilibration_evidence="brain and plasma half-lives are equal",
    )
    assert evidenced.assumes_rapid_equilibrium_unverified is False


# ---------------------------------------------------------------------------
# 2. Wiring: pk.py and engine.py
# ---------------------------------------------------------------------------


def _two_node_network():
    return scenarios.reward_circuit_network()


def _single_ligand_site(kon=0.1, koff=0.05):
    return ReceptorSite(
        name="site", ligands=[Ligand(name="drug", kon=kon, koff=koff, efficacy=1.0)]
    )


def _sim(pk_model, network=None):
    network = network or _two_node_network()
    return Simulation(
        network=network,
        receptor_systems=[
            ReceptorSystem(
                pk_models=[pk_model],
                receptor_site=_single_ligand_site(),
                coupling={network.nodes[0].name: 1.0},
            )
        ],
    )


def test_no_partition_is_exactly_the_pre_p1_5_behavior():
    """A PK model with no brain partition must be indistinguishable from the
    pre-P1.5 code: factor exactly 1.0 (not approximately), and the receptor
    concentration the same array of values as the plasma concentration."""
    model = OneCompartmentPK(ke=0.05)
    assert model.brain_partition is None
    assert model.receptor_concentration_factor == 1.0
    assert model.receptor_compartment.startswith("total plasma")
    state = np.array([7.0])
    assert model.receptor_concentration(state) == model.concentration(state)

    sim = _sim(model)
    assert np.array_equal(sim.receptor_concentration_factors, np.ones(1))
    result = sim.run(doses=[10.0], t_span=(0.0, 20.0), t_eval=np.linspace(0, 20, 21))
    # Identity of objects, not merely of values: with no partition anywhere
    # the engine hands the plasma array straight to the receptor layer.
    assert result["receptor_concentrations"] is result["concentrations"]
    assert result["plasma_concentrations"] is result["concentrations"]


def test_partition_scales_the_concentration_the_receptor_binds():
    """The acceptance criterion, stated as an equivalence.

    A partitioned model dosed at C0 and an unpartitioned model dosed at
    factor*C0 have the same unbound brain concentration at every instant, so
    they must produce identical occupancy and identical firing rates - while
    their *plasma* concentrations differ by exactly the factor. If occupancy
    were still being driven by plasma, this test would fail.
    """
    factor = 0.18
    partition = BrainPartition(
        drug="drug",
        species="rat",
        basis="Kp,uu",
        ratio=0.9,
        fu_plasma=0.2,
        source="synthetic test partition",
    )
    assert partition.unbound_brain_factor == pytest.approx(factor)

    brain_sim = _sim(OneCompartmentPK(ke=0.05, brain_partition=partition))
    plasma_sim = _sim(OneCompartmentPK(ke=0.05))

    t_eval = np.linspace(0.0, 40.0, 41)
    brain = brain_sim.run(doses=[100.0], t_span=(0.0, 40.0), t_eval=t_eval)
    plasma = plasma_sim.run(
        doses=[100.0 * factor], t_span=(0.0, 40.0), t_eval=t_eval
    )

    assert np.allclose(brain["occupancy"], plasma["occupancy"], rtol=0, atol=1e-12)
    assert np.allclose(brain["rates"], plasma["rates"], rtol=0, atol=1e-12)
    assert np.allclose(
        brain["receptor_concentrations"], plasma["concentrations"], atol=1e-12
    )
    # ... and the plasma traces really are different, so the equality above is
    # not vacuous.
    assert brain["concentrations"][0, 0] == pytest.approx(100.0)
    assert plasma["concentrations"][0, 0] == pytest.approx(18.0)


def test_partition_lowers_occupancy_when_the_factor_is_below_one():
    """Direction check, separate from the equivalence above: for diazepam the
    unbound brain factor is 0.18, so driving with brain rather than plasma
    must *reduce* occupancy. The pre-P1.5 code was overstating it."""
    partition = drug_pk.diazepam_rat_brain_partition()
    network = _two_node_network()
    t_eval = np.linspace(0.0, 60.0, 61)

    with_brain = _sim(
        OneCompartmentPK(ke=0.013, brain_partition=partition), network
    ).run(doses=[100.0], t_span=(0.0, 60.0), t_eval=t_eval)
    plasma_only = _sim(OneCompartmentPK(ke=0.013), network).run(
        doses=[100.0], t_span=(0.0, 60.0), t_eval=t_eval
    )
    assert with_brain["occupancy"].max() < plasma_only["occupancy"].max()


def test_mixed_partitioned_and_unpartitioned_ligands_in_one_simulation():
    """The engine holds one factor per ligand, so a system where only some
    ligands have a sourced partition must work and must apply the factor
    ligand-wise, not globally."""
    partition = BrainPartition(
        drug="a",
        species="rat",
        basis="Kp",
        ratio=4.0,
        fu_brain=0.05,
        source="synthetic test partition",
    )
    network = _two_node_network()
    site = ReceptorSite(
        name="site",
        ligands=[
            Ligand(name="a", kon=0.1, koff=0.05, efficacy=1.0),
            Ligand(name="b", kon=0.1, koff=0.05, efficacy=0.0),
        ],
    )
    sim = Simulation(
        network=network,
        receptor_systems=[
            ReceptorSystem(
                pk_models=[
                    OneCompartmentPK(ke=0.05, brain_partition=partition),
                    OneCompartmentPK(ke=0.05),
                ],
                receptor_site=site,
                coupling={network.nodes[0].name: 1.0},
            )
        ],
    )
    assert sim.receptor_concentration_factors.tolist() == [0.2, 1.0]
    result = sim.run(
        doses=[10.0, 10.0], t_span=(0.0, 10.0), t_eval=np.linspace(0, 10, 11)
    )
    assert np.allclose(
        result["receptor_concentrations"][0], 0.2 * result["concentrations"][0]
    )
    assert np.array_equal(
        result["receptor_concentrations"][1], result["concentrations"][1]
    )


def test_attaching_an_unusable_partition_fails_at_construction_not_at_runtime():
    """A Kp-only partition must be rejected when the PK model is built, with
    the drug's own citation in the message, rather than producing a silently
    wrong occupancy trace or an exception thousands of RHS calls later."""
    with pytest.raises(UnusablePartitionError) as excinfo:
        OneCompartmentPK(
            ke=0.05, brain_partition=drug_pk.haloperidol_rat_brain_partition()
        )
    assert "Sunderland1987" in str(excinfo.value)


def test_a_pk_model_refuses_a_partition_from_another_species():
    with pytest.raises(ValueError) as excinfo:
        drug_pk.diazepam_rat_ip_pk(
            BODY_WEIGHT_KG,
            brain_partition=drug_pk.diazepam_mouse_brain_partition(),
        )
    assert "cross-species" in str(excinfo.value)


def test_brain_partition_must_be_a_brain_partition():
    with pytest.raises(TypeError):
        OneCompartmentPK(ke=0.05, brain_partition=4.5)


# ---------------------------------------------------------------------------
# 3. The drug numbers
# ---------------------------------------------------------------------------


def test_diazepam_rat_partition_is_the_documented_derivation():
    p = drug_pk.diazepam_rat_brain_partition()
    assert p.basis == "Kp"
    assert p.ratio == drug_pk.DIAZEPAM_RAT_IP_BRAIN_PLASMA_RATIO == 4.5
    assert p.fu_plasma == drug_pk.DIAZEPAM_RAT_IP_FREE_FRACTION == 0.196
    assert p.fu_brain == drug_pk.DIAZEPAM_RAT_FU_BRAIN_WHOLE == 0.040
    assert p.kp_uu == pytest.approx(4.5 * 0.040 / 0.196)
    assert p.kp_uu == pytest.approx(0.918, abs=0.001)
    assert p.unbound_brain_factor == pytest.approx(0.18)
    assert p.can_drive_receptor is True


def test_diazepam_rat_kp_uu_agrees_with_an_independent_measurement():
    """[Friden2009] measured rat diazepam Kp,uu = 1.07 directly; the value
    derived here from [Friedman1986] + [Gustafsson2019] is 0.918. Those are
    two different laboratories, two different methods and two different
    decades, so the tolerance below is what the agreement actually is (14%),
    not a target - it is asserted as an upper bound so that a later edit that
    made the derivation worse would fail this test.

    Worth reading next to the *Kp* comparison in the following test: the two
    studies' Kp values disagree twice as badly as their Kp,uu values, which is
    the behavior the Kp/Kp,uu distinction predicts.
    """
    derived = drug_pk.diazepam_rat_brain_partition().kp_uu
    measured = drug_pk.DIAZEPAM_RAT_KP_UU_FRIDEN2009
    relative_error = abs(derived - measured) / measured
    assert relative_error < 0.20, relative_error


def test_the_two_rat_diazepam_kp_values_disagree_and_that_is_recorded():
    """4.5 [Friedman1986] against 2.28 [Friden2009]: a factor of ~2, which is
    left on the record rather than averaged away. Pinned as a test so the
    disagreement cannot quietly disappear."""
    assert drug_pk.DIAZEPAM_RAT_IP_BRAIN_PLASMA_RATIO == 4.5
    assert drug_pk.DIAZEPAM_RAT_KP_FRIDEN2009 == 2.28
    ratio = drug_pk.DIAZEPAM_RAT_IP_BRAIN_PLASMA_RATIO / drug_pk.DIAZEPAM_RAT_KP_FRIDEN2009
    assert 1.8 < ratio < 2.1
    # The 2-fold gap in Kp is larger than the gap in Kp,uu (previous test).
    # That ordering is the substantive claim, not the individual numbers.
    kp_uu_gap = abs(
        drug_pk.diazepam_rat_brain_partition().kp_uu
        - drug_pk.DIAZEPAM_RAT_KP_UU_FRIDEN2009
    ) / drug_pk.DIAZEPAM_RAT_KP_UU_FRIDEN2009
    kp_gap = abs(
        drug_pk.DIAZEPAM_RAT_IP_BRAIN_PLASMA_RATIO - drug_pk.DIAZEPAM_RAT_KP_FRIDEN2009
    ) / drug_pk.DIAZEPAM_RAT_KP_FRIDEN2009
    assert kp_uu_gap < kp_gap


def test_regional_fu_brain_is_not_homogeneous_and_whole_brain_is_the_default():
    """[Gustafsson2019]'s finding is that brain tissue binding varies by
    region; the default partition uses the whole-brain value, and the spread
    is the error that default carries into a non-regional model."""
    by_region = drug_pk.DIAZEPAM_RAT_FU_BRAIN_BY_REGION
    assert by_region["whole_brain"] == drug_pk.DIAZEPAM_RAT_FU_BRAIN_WHOLE
    regional = [v for k, v in by_region.items() if k != "whole_brain"]
    assert max(regional) / min(regional) > 1.4
    assert drug_pk.diazepam_rat_brain_partition().region == "whole_brain"
    cerebellum = drug_pk.diazepam_rat_brain_partition(region="cerebellum")
    assert cerebellum.fu_brain == by_region["cerebellum"]
    assert cerebellum.unbound_brain_factor > (
        drug_pk.diazepam_rat_brain_partition().unbound_brain_factor
    )
    with pytest.raises(KeyError):
        drug_pk.diazepam_rat_brain_partition(region="thalamus")


@pytest.mark.parametrize(
    "builder, reported",
    [
        (drug_pk.diazepam_mouse_brain_partition, drug_pk.DIAZEPAM_MOUSE_KP_UU_REPORTED),
        (
            drug_pk.haloperidol_mouse_brain_partition,
            drug_pk.HALOPERIDOL_MOUSE_KP_UU_REPORTED,
        ),
    ],
)
def test_mouse_partitions_are_internally_consistent(builder, reported):
    """[Doran2005] and [Maurer2005] are both closed access, so their values
    reach this project only as quotes in [Ma2024]'s Table 1. The one check
    available on a second-hand quote is whether the quoted Kp, fu_p and fu_b
    reproduce the quoted Kp,uu; they do, to within the table's rounding.
    A transcription error in any of the four would break this."""
    derived = builder().kp_uu
    assert derived == pytest.approx(reported, rel=0.02)


def test_mouse_partitions_are_recorded_but_not_attachable_to_anything():
    """A partition is not a licence to invent the kinetics it multiplies:
    there is no mouse PK for either drug, so these exist as data only."""
    for builder in (
        drug_pk.diazepam_mouse_brain_partition,
        drug_pk.haloperidol_mouse_brain_partition,
    ):
        assert builder().species == "mouse"
        assert builder().can_drive_receptor is True
    # Complete on paper, and there is still no mouse PK for either drug.
    with pytest.raises(drug_pk.UnsourcedParameterError):
        drug_pk.diazepam_mouse_pk("IP", 0.025)


@pytest.mark.parametrize("drug", ["haloperidol", "fentanyl"])
def test_total_kp_only_drugs_are_flagged_and_explained(drug):
    """Haloperidol and fentanyl each have a cited rat brain:plasma ratio and
    no rat brain unbound fraction, so they can report a total brain
    concentration and may not drive occupancy. The status and the search
    record must both say so."""
    status, explanation = drug_pk.brain_partition_status(drug, "rat")
    assert status == "total-brain-only"
    assert "f(u,brain)" in explanation
    partition = drug_pk.brain_partition_for(
        drug, "rat", require_receptor_drivable=False
    )
    assert partition.basis == "Kp"
    assert partition.can_drive_receptor is False
    assert partition.total_brain_factor > 1.0
    record = drug_pk.SEARCHED_AND_EMPTY_BRAIN_PARTITION[(drug, "rat", "fu_brain")]
    assert "Searched (2026-08)" in record
    assert "unusable" in record


def test_haloperidol_and_fentanyl_rat_models_say_they_are_plasma_driven():
    """The refusal must not be silent. These two models build fine - their
    kinetics are sourced - and report which compartment they drive."""
    for model in (
        drug_pk.haloperidol_rat_iv_pk(BODY_WEIGHT_KG),
        drug_pk.fentanyl_rat_iv_pk(BODY_WEIGHT_KG),
    ):
        assert model.brain_partition is None
        assert model.receptor_concentration_factor == 1.0
        assert model.receptor_compartment == (
            "total plasma (no brain partition attached)"
        )


def test_diazepam_rat_models_drive_the_receptor_with_unbound_brain_by_default():
    for model in (
        drug_pk.diazepam_rat_iv_pk(BODY_WEIGHT_KG),
        drug_pk.diazepam_rat_ip_pk(BODY_WEIGHT_KG),
        drug_pk.diazepam_rat_pk("IP", BODY_WEIGHT_KG),
    ):
        assert model.receptor_compartment == "unbound brain"
        assert model.receptor_concentration_factor == pytest.approx(0.18)
        assert model.species == "rat"


def test_the_partition_does_not_touch_the_plasma_curve():
    """P1.5 must add a layer, not perturb P1.1-P1.4. The plasma
    concentration-time course of the rat diazepam IP model has to be
    bit-identical with and without a partition attached."""
    with_partition = drug_pk.diazepam_rat_ip_pk(BODY_WEIGHT_KG)
    without = drug_pk.diazepam_rat_ip_pk(BODY_WEIGHT_KG, brain_partition=None)
    assert without.brain_partition is None
    state = with_partition.initial_state(2.0)
    assert np.array_equal(state, without.initial_state(2.0))
    assert np.array_equal(
        with_partition.concentration_rhs(state), without.concentration_rhs(state)
    )
    assert with_partition.terminal_half_life_min == without.terminal_half_life_min
    assert with_partition.auc_0_inf(2.0) == without.auc_0_inf(2.0)


def test_fentanyl_figure_reads_are_internally_consistent():
    """The fentanyl Kp is read off a bar chart, so it gets a consistency check
    the printed numbers do not need: panel A (brain ng/g) divided by panel B
    (brain:plasma) has to reproduce a 1 h plasma concentration compatible with
    the paper's own printed Cmax and half-life.

    One-sided by construction - the printed-PK estimate ignores the
    absorption phase still contributing at 1 h and is therefore biased low -
    so the assertion is agreement within 1.5-fold, in the expected direction.
    """
    from_figures, from_printed, ratio = (
        drug_pk.fentanyl_rat_brain_plasma_figure_read_cross_check()
    )
    assert from_figures > from_printed  # the stated direction of the bias
    assert ratio < 1.5, (from_figures, from_printed, ratio)


def test_fentanyl_partition_is_labeled_a_figure_read():
    p = drug_pk.fentanyl_rat_brain_partition()
    assert "FIGURE READ" in p.source
    assert drug_pk.FENTANYL_RAT_KP_FIGURE_READ_UNCERTAINTY > 0
    per_region = drug_pk.FENTANYL_RAT_KP_BY_REGION_FIGURE_READ
    assert p.ratio == pytest.approx(sum(per_region.values()) / len(per_region))
    assert 15.0 <= min(per_region.values()) and max(per_region.values()) <= 18.0
    assert drug_pk.fentanyl_rat_brain_partition(region="medulla").ratio == 18.0
    with pytest.raises(KeyError):
        drug_pk.fentanyl_rat_brain_partition(region="cerebellum")


def test_rapid_equilibrium_is_evidenced_for_the_two_drugs_that_have_evidence():
    """The instantaneous-partition assumption is only legitimate when brain
    equilibration is fast. Diazepam and fentanyl each have a published
    observation that it is; haloperidol does not, and that gap is visible."""
    diazepam = drug_pk.diazepam_rat_brain_partition()
    assert diazepam.assumes_rapid_equilibrium_unverified is False
    assert "0.89 h" in diazepam.equilibration_evidence

    fentanyl = drug_pk.fentanyl_rat_brain_partition()
    assert fentanyl.assumes_rapid_equilibrium_unverified is False
    assert "HugMurphy1981" in fentanyl.equilibration_evidence

    haloperidol = drug_pk.haloperidol_rat_brain_partition()
    assert haloperidol.assumes_rapid_equilibrium_unverified is True


def test_the_human_gap_is_recorded_and_the_legacy_scenarios_stay_plasma_driven():
    """The calibrated scenarios run on HUMAN clinical PK, and no human Kp,uu
    was sourced, so P1.5 deliberately left them alone. Both halves of that
    are asserted here: the search record exists, and every calibrated
    scenario's PK models are still plasma-driven with factor exactly 1.0."""
    record = drug_pk.SEARCHED_AND_EMPTY_BRAIN_PARTITION[("*", "human", "*")]
    assert "postmortem" in record
    assert "rule 2" in record or "cross-species" in record

    for builder in (
        scenarios.mu_opioid_pk_models_calibrated,
        scenarios.gaba_a_pk_models_calibrated,
        scenarios.d2_pk_models_calibrated,
    ):
        for model in builder():
            assert model.brain_partition is None
            assert model.receptor_concentration_factor == 1.0


def test_no_unsourced_partition_constant_can_be_added_quietly():
    """Guard in the style of `test_fentanyl_pk.py`'s bioavailability guard.

    Every module-level partition constant must be one of the ones P1.5
    sourced. Adding, say, `FENTANYL_RAT_FU_BRAIN = 0.02` because it "looks
    right for a lipophilic base" would make the fentanyl partition
    receptor-drivable and would silently change every fentanyl occupancy
    result; this test makes that require a deliberate edit here, with a
    citation and a docs/architecture.md update in the same commit.
    """
    allowed = {
        "DIAZEPAM_RAT_FU_BRAIN_WHOLE",
        "DIAZEPAM_RAT_FU_BRAIN_WHOLE_SD",
        "DIAZEPAM_RAT_FU_BRAIN_BY_REGION",
        "DIAZEPAM_RAT_KP_FRIDEN2009",
        "DIAZEPAM_RAT_KP_UU_FRIDEN2009",
        "DIAZEPAM_MOUSE_KP",
        "DIAZEPAM_MOUSE_KP_UU_REPORTED",
        "DIAZEPAM_MOUSE_FU_PLASMA",
        "DIAZEPAM_MOUSE_FU_BRAIN",
        "HALOPERIDOL_RAT_KP",
        "HALOPERIDOL_MOUSE_KP",
        "HALOPERIDOL_MOUSE_KP_UU_REPORTED",
        "HALOPERIDOL_MOUSE_FU_PLASMA",
        "HALOPERIDOL_MOUSE_FU_BRAIN",
        "FENTANYL_RAT_KP_BY_REGION_FIGURE_READ",
        "FENTANYL_RAT_KP_FIGURE_READ_UNCERTAINTY",
        "FENTANYL_RAT_KP_REGION_MEAN_FIGURE_READ",
    }
    for name in dir(drug_pk):
        if not name.isupper():
            continue
        if "_KP" in name or "FU_BRAIN" in name or "FU_PLASMA" in name:
            assert name in allowed, (
                "%s is a partition/unbound-fraction constant that P1.5 did "
                "not source; adding one needs a citation here and in "
                "docs/architecture.md" % name
            )


# ---------------------------------------------------------------------------
# End to end: mg/kg -> plasma -> unbound brain -> occupancy
# ---------------------------------------------------------------------------


def test_dose_to_brain_concentration_to_occupancy_end_to_end():
    """The whole Phase 1 chain for the one drug that has every link.

    2 mg/kg diazepam IP in a 250 g rat, through [Friedman1986]+[Hironaka1984]
    kinetics, through the [Friedman1986]+[Gustafsson2019] partition, into the
    [Mohler1977]-calibrated benzodiazepine site, inside one coupled ODE
    system. Checked against closed forms at every link rather than against
    itself.
    """
    body_weight_kg = 0.25
    dose_mg_per_kg = 2.0
    pk_model = drug_pk.diazepam_rat_ip_pk(body_weight_kg)
    diazepam_ligand = scenarios.gaba_a_benzodiazepine_ligands_calibrated()[0]
    assert diazepam_ligand.name == "diazepam"

    network = scenarios.amygdala_anxiety_circuit_network()
    site = ReceptorSite(name="benzodiazepine_site", ligands=[diazepam_ligand])
    sim = Simulation(
        network=network,
        receptor_systems=[
            ReceptorSystem(
                pk_models=[pk_model],
                receptor_site=site,
                coupling={},
                connection_modulation=scenarios.gaba_a_connection_modulation_calibrated(),
            )
        ],
    )

    t_eval = np.linspace(0.0, 180.0, 181)
    result = sim.run(
        doses=[dose_mg_per_kg], t_span=(0.0, 180.0), t_eval=t_eval
    )

    plasma = result["concentrations"][0]
    brain = result["receptor_concentrations"][0]

    # Link 1, dose -> plasma: the depot starts at F*Dose/V, in nM, and the
    # peak plasma concentration is strictly below it (absorption is never
    # instantaneous).
    depot0 = pk_model.dose_equivalent_concentration(dose_mg_per_kg)
    assert depot0 == pytest.approx(
        dose_mg_per_kg
        / drug_pk.DIAZEPAM_RAT_IP_VD_OVER_F_L_PER_KG
        * 1.0e6
        / drug_pk.DIAZEPAM_MOLAR_MASS_G_PER_MOL
    )
    assert 0.0 < plasma.max() < depot0

    # Link 2, plasma -> unbound brain: exactly fu_brain * Kp, everywhere.
    assert np.allclose(brain, 0.18 * plasma, rtol=0, atol=1e-12)
    assert brain.max() < plasma.max()  # the factor is below 1 for diazepam

    # Link 3, unbound brain -> occupancy: at the concentration peak the
    # binding kinetics are fast enough (koff ~ 0.36/min against a ~50 min
    # plasma half-life) that occupancy is close to its instantaneous
    # equilibrium value C/(C + Kd) with Kd = koff/kon.
    peak = int(np.argmax(brain))
    kd = diazepam_ligand.koff / diazepam_ligand.kon
    quasi_equilibrium = brain[peak] / (brain[peak] + kd)
    assert result["occupancy"][0][peak] == pytest.approx(quasi_equilibrium, rel=0.05)

    # And the occupancy that plasma would have produced is materially higher,
    # which is the quantitative content of P1.5 for this drug.
    plasma_driven = plasma[peak] / (plasma[peak] + kd)
    assert plasma_driven > result["occupancy"][0][peak]
