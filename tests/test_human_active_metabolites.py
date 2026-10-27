"""P21.4: metabolites that occupy the same receptor as their parent."""

import math
from dataclasses import replace

import pytest

from neurocascade.human.active_metabolites import (
    ACTIVE_METABOLITES,
    ARIPIPRAZOLE_DEHYDRO_ARIPIPRAZOLE,
    GRUNDER_2008_ARIPIPRAZOLE_EC50_NG_PER_ML,
    HALOPERIDOL_REDUCED_HALOPERIDOL,
    RISPERIDONE_9_HYDROXYRISPERIDONE,
    ActiveMetabolite,
    active_metabolites_for,
    equipotency_check,
    exposure_ratio_from_ec50_bases,
    moiety_ec50_from_components,
    registered_parent_drugs,
    require_parent_only_path,
)
from neurocascade.human.dose_to_occupancy import predict_occupancy
from neurocascade.human.receptor_parameters import require_human_parameter

#: The chain's own held-out mean absolute error on the V1 validated claim. Every
#: "small enough to ignore" judgement in this module is measured against it
#: rather than against a round number chosen here.
V1_HELD_OUT_MAE = 0.0351


# ---------------------------------------------------------------------------
# The acceptance criterion: the parent-only path is closed for these drugs
# ---------------------------------------------------------------------------


def _receptor_record_for(drug, summed):
    """The registered D2 record relabelled onto another drug.

    Relabelling rather than inventing: the point of the test is the drug
    identity and the analyte flag, and reusing the real record keeps every other
    field a sourced one so that nothing else in the chain can be the reason the
    call fails.
    """
    record = require_human_parameter("D2")
    row = replace(record.ec50_record, drug=drug, analyte_is_summed=summed)
    return replace(record, ligand=drug, ec50_record=row)


@pytest.mark.parametrize(
    "drug, metabolite",
    [
        ("risperidone", "9-hydroxyrisperidone"),
        ("aripiprazole", "dehydro-aripiprazole"),
    ],
)
def test_a_drug_with_an_active_metabolite_cannot_be_predicted_parent_only(
    drug, metabolite
):
    """The acceptance criterion for P21.4, exercised through the real chain.

    The record handed in is deliberately flagged ``analyte_is_summed=False``, so
    the pre-existing summed-moiety gate cannot be what rejects it. Only the
    active-metabolite gate can, and the message has to name the metabolite: a
    refusal that does not say which molecule is missing is a refusal the caller
    will work around.
    """
    record = _receptor_record_for(drug, summed=False)
    with pytest.raises(ValueError) as excinfo:
        predict_occupancy(5.0, receptor_record=record)
    message = str(excinfo.value)
    assert metabolite in message
    assert "parent-only path" in message
    # Not the generic "this chain is haloperidol's" rejection, which would send
    # the caller off to generalise the chain and reopen the same hole.
    assert "is for %r, not" % drug not in message


def test_the_metabolite_gate_fires_before_the_summed_analyte_gate():
    """Ordering matters, because the two refusals recommend opposite fixes.

    The summed-moiety message says the constant is on the wrong axis, which
    invites finding a parent-only constant. For risperidone that fix does not
    exist, and following it would waste the search P21.2 already ran.
    """
    record = _receptor_record_for("risperidone", summed=True)
    with pytest.raises(ValueError, match="9-hydroxyrisperidone"):
        predict_occupancy(5.0, receptor_record=record)


def test_require_parent_only_path_refuses_with_a_recorded_reason():
    for drug in ("risperidone", "aripiprazole"):
        with pytest.raises(ValueError) as excinfo:
            require_parent_only_path(drug)
        message = str(excinfo.value)
        record = ACTIVE_METABOLITES[drug][0]
        assert record.reason in message
        assert record.exposure_citation in message


def test_an_unregistered_drug_is_not_silently_cleared():
    """Absence from the registry returns an empty tuple, not an assurance.

    The distinction is the whole reason the gate returns records instead of a
    boolean: an empty result means nobody has looked, and the caller can see
    that it is empty.
    """
    assert require_parent_only_path("olanzapine") == ()
    assert active_metabolites_for("olanzapine") == ()


# ---------------------------------------------------------------------------
# Haloperidol: cleared by a bound, and the bound is checked independently
# ---------------------------------------------------------------------------


def test_haloperidol_is_cleared_and_the_chain_still_runs():
    records = require_parent_only_path("haloperidol")
    assert [r.disposition for r in records] == ["bounded"]
    prediction = predict_occupancy(5.0)
    assert prediction.provenance == "measured"
    assert len(prediction.metabolite_caveats) == 1
    assert prediction.metabolite_caveats[0].metabolite == "reduced haloperidol"
    assert "reduced haloperidol" in prediction.summary()
    # The metabolite is a caveat, not a layer: nothing was consumed from it, so
    # it must not appear among the things the numbers rest on.
    assert all("metabolite" not in layer.name for layer in prediction.layers)


def test_the_haloperidol_bound_is_below_the_chains_own_validated_error():
    record = HALOPERIDOL_REDUCED_HALOPERIDOL
    assert record.drive_ratio() == pytest.approx(1.04 / 85.0)
    assert record.max_occupancy_error() < V1_HELD_OUT_MAE / 10.0


def test_the_bound_survives_the_whole_observed_exposure_range():
    """Uematsu's ratio ranged 0.39 to 1.99 across ten patients.

    A bound computed at the mean would be worthless if the upper end of the
    observed range broke it, so the upper end is what is tested.
    """
    worst = replace(HALOPERIDOL_REDUCED_HALOPERIDOL, exposure_ratio=1.99)
    assert worst.max_occupancy_error() < V1_HELD_OUT_MAE


def test_the_cross_species_conditional_is_carried_by_the_object():
    assert HALOPERIDOL_REDUCED_HALOPERIDOL.affinity_species == "rat"
    assert HALOPERIDOL_REDUCED_HALOPERIDOL.affinity_is_cross_species
    summary = HALOPERIDOL_REDUCED_HALOPERIDOL.summary()
    assert "CROSS-SPECIES" in summary
    prediction = predict_occupancy(5.0)
    assert "conditional on the rat affinity ratio" in prediction.summary()


# ---------------------------------------------------------------------------
# The arithmetic, checked against an independent derivation
# ---------------------------------------------------------------------------


def _numerical_max_occupancy_error(d, samples=200001):
    """Maximise the two-ligand minus parent-only difference by brute force.

    Deliberately not the closed form under test. The grid is on log-concentration
    so that the maximum, which sits at ``x = 1/sqrt(1+d)``, is not straddled by a
    coarse linear spacing at small ``d``.
    """
    g = 1.0 + d
    best = 0.0
    for i in range(samples):
        log_x = -6.0 + 12.0 * i / (samples - 1)
        x = math.exp(log_x)
        best = max(best, g * x / (1.0 + g * x) - x / (1.0 + x))
    return best


@pytest.mark.parametrize("d", [0.001, 0.0122, 0.1, 0.4, 1.0, 5.0])
def test_the_closed_form_bound_matches_a_brute_force_maximisation(d):
    record = replace(
        HALOPERIDOL_REDUCED_HALOPERIDOL, exposure_ratio=d, affinity_ratio=1.0
    )
    assert record.max_occupancy_error() == pytest.approx(
        _numerical_max_occupancy_error(d), rel=1e-4
    )


def test_the_apparent_ec50_factor_is_the_hyperbola_the_parent_only_fit_sees():
    """A parent-only fit to two-ligand data recovers ``K_p / (1 + d)``.

    Checked by construction rather than by algebra: generate occupancy from the
    two-ligand form, then solve for the single constant that reproduces it.
    """
    record = HALOPERIDOL_REDUCED_HALOPERIDOL
    d = record.drive_ratio()
    k_p = 1.8849
    for concentration in (0.1, 1.0, 5.0, 50.0):
        true_occupancy = (
            concentration * (1.0 + d) / k_p
        ) / (1.0 + concentration * (1.0 + d) / k_p)
        apparent_ec50 = concentration * (1.0 - true_occupancy) / true_occupancy
        assert apparent_ec50 == pytest.approx(
            k_p / record.apparent_ec50_factor(), rel=1e-12
        )


def test_a_moiety_ec50_lies_between_its_components():
    low = moiety_ec50_from_components(4.0, 9.0, 1.0)
    high = moiety_ec50_from_components(4.0, 9.0, 0.0)
    assert low == pytest.approx(4.0)
    assert high == pytest.approx(9.0)
    for fraction in (0.1, 0.5, 0.9):
        value = moiety_ec50_from_components(4.0, 9.0, fraction)
        assert 4.0 < value < 9.0
    # Equipotent components give the same constant at every composition.
    assert moiety_ec50_from_components(7.0, 7.0, 0.3) == pytest.approx(7.0)


def test_the_moiety_helper_rejects_impossible_inputs():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        moiety_ec50_from_components(4.0, 9.0, 1.5)
    with pytest.raises(ValueError, match="must be positive"):
        moiety_ec50_from_components(0.0, 9.0, 0.5)


# ---------------------------------------------------------------------------
# Two EC50 bases from one study, and what they measure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k_p, k_m", [(1.0, 1.0), (0.4, 9.0), (9.0, 0.4), (3.3, 3.3)])
@pytest.mark.parametrize("r", [0.05, 0.4, 1.0, 7.5])
def test_the_two_basis_identity_recovers_the_exposure_ratio_without_affinities(
    k_p, k_m, r
):
    """``M*/P* - 1 = r`` exactly, for any pair of affinities.

    Built from the two-ligand form rather than from the identity: solve for the
    parent concentration at half occupancy, form the moiety at that same point,
    and hand both to the helper. The affinities are swept over a 22-fold range in
    both directions precisely because the claim is that they cancel. If they did
    not, this is where it would show.
    """
    parent_ec50 = 1.0 / (1.0 / k_p + r / k_m)
    moiety_ec50 = parent_ec50 * (1.0 + r)
    # Sanity: the constructed point really is half occupancy in the two-ligand form.
    drive = parent_ec50 / k_p + r * parent_ec50 / k_m
    assert drive / (1.0 + drive) == pytest.approx(0.5, rel=1e-12)
    assert exposure_ratio_from_ec50_bases(parent_ec50, moiety_ec50) == pytest.approx(
        r, rel=1e-12
    )


def test_the_two_basis_helper_rejects_a_moiety_below_its_own_parent():
    with pytest.raises(ValueError, match="not above the parent"):
        exposure_ratio_from_ec50_bases(10.0, 9.0)
    with pytest.raises(ValueError, match="not above the parent"):
        exposure_ratio_from_ec50_bases(10.0, 10.0)
    with pytest.raises(ValueError, match="must be positive"):
        exposure_ratio_from_ec50_bases(10.0, -1.0)


def test_grunders_two_bases_disagree_with_the_labels_forty_percent():
    """The one place aripiprazole's exposure ratio can be measured, it is not 0.40.

    Both of Gruender's regions give 1.0. The reviews print the constants as
    integers with no dispersion, so the check that matters is whether rounding
    alone could reconcile them with the label: it cannot, because even the
    rounding interval most favourable to the label stays above 0.84.
    """
    ratios = {
        region: exposure_ratio_from_ec50_bases(parent, moiety)
        for region, (parent, moiety) in
        GRUNDER_2008_ARIPIPRAZOLE_EC50_NG_PER_ML.items()
    }
    assert set(ratios) == {"putamen", "caudate"}
    for ratio in ratios.values():
        assert ratio == pytest.approx(1.0)

    label_ratio = ARIPIPRAZOLE_DEHYDRO_ARIPIPRAZOLE.exposure_ratio
    assert label_ratio == 0.40
    for parent, moiety in GRUNDER_2008_ARIPIPRAZOLE_EC50_NG_PER_ML.values():
        lowest = exposure_ratio_from_ec50_bases(parent + 0.5, moiety - 0.5)
        assert lowest > 0.84
        assert lowest > 2.0 * label_ratio

    # And the disagreement is the size that matters: on the label's ratio the
    # metabolite is a 2.4x-MAE problem, on Gruender's it is a 4.9x one, and the
    # record is refused rather than bounded because the sources do not say which.
    illustrative = replace(
        ARIPIPRAZOLE_DEHYDRO_ARIPIPRAZOLE, affinity_ratio=1.0, affinity_species="human"
    )
    on_label = illustrative.max_occupancy_error()
    on_grunder = replace(illustrative, exposure_ratio=1.0).max_occupancy_error()
    assert on_label == pytest.approx(0.0839, abs=0.0001)
    assert on_grunder == pytest.approx(0.1716, abs=0.0001)
    assert on_grunder / on_label > 2.0
    assert min(on_label, on_grunder) > 2.0 * V1_HELD_OUT_MAE


# ---------------------------------------------------------------------------
# Risperidone: the refusal is priced, not argued
# ---------------------------------------------------------------------------


def test_the_risperidone_refusal_costs_more_than_sixteen_times_the_chains_error():
    """Both ratios exist for this pair, so the cost of ignoring the metabolite
    is a number rather than a worry.

    Lenk et al.'s normal-metaboliser MPR against Schotte's same-assay human D2L Ki
    pair gives d = 12.2. The same closed form that clears haloperidol at 0.0030
    condemns risperidone at 0.57, which is the quantitative form of the brief's
    "understate it silently".
    """
    record = RISPERIDONE_9_HYDROXYRISPERIDONE
    assert record.drive_ratio() == pytest.approx(9.93 / (4.8 / 5.9), rel=1e-12)
    assert record.drive_ratio() == pytest.approx(12.206, abs=0.001)
    assert record.max_occupancy_error() == pytest.approx(0.5684, abs=0.0001)
    assert record.max_occupancy_error() > 16.0 * V1_HELD_OUT_MAE
    # It is refused, not bounded, and the bound is what says why.
    assert record.disposition == "refused"
    assert (
        record.max_occupancy_error()
        > 100.0 * HALOPERIDOL_REDUCED_HALOPERIDOL.max_occupancy_error()
    )


def test_the_risperidone_conclusion_does_not_rest_on_the_affinity_ratio():
    """An in-vitro Ki ratio is the weakest input here, so the result is stressed
    against it rather than defended.

    Even at an affinity ratio ten times more favourable to the parent than the
    measured one, ignoring the metabolite still costs nearly six times the
    chain's held-out error. It is the exposure ratio near 10 that decides this
    drug, and that one is measured in 243 patients.
    """
    record = RISPERIDONE_9_HYDROXYRISPERIDONE
    stressed = replace(record, affinity_ratio=record.affinity_ratio * 10.0)
    assert stressed.drive_ratio() == pytest.approx(1.2205, abs=0.0001)
    assert stressed.max_occupancy_error() == pytest.approx(0.19683, abs=0.0001)
    assert stressed.max_occupancy_error() > 5.0 * V1_HELD_OUT_MAE

    # The lowest metabolite:parent ratio anywhere in Lenk et al.'s table, from poor
    # metabolisers, is the only phenotype where the parent-only path would be
    # anywhere near defensible, and the record's spread says so.
    poorest = replace(record, exposure_ratio=0.25)
    assert poorest.max_occupancy_error() < V1_HELD_OUT_MAE * 3.0
    assert "0.25" in record.exposure_spread
    assert "33.11" in record.exposure_spread


# ---------------------------------------------------------------------------
# The external comparison
# ---------------------------------------------------------------------------


def test_the_equipotency_prediction_is_recorded_with_its_miss():
    """Arakawa's paliperidone ED50 predicts Remington's risperidone moiety ED50.

    Two different groups, two different drugs, two different papers, and no
    number on either side chosen in this repository. The prediction misses the
    long-acting-injectable value by 1.70-fold. That is asserted as the result,
    not tuned away: the ledger records it as a failed check with the four
    candidate explanations attached.
    """
    check = equipotency_check(parent_fraction_of_moiety=0.5)
    assert check.predicted_ng_per_mL == pytest.approx(6.523, abs=0.001)
    assert check.measured_ng_per_mL == 11.06
    assert check.fold_error == pytest.approx(1.696, abs=0.001)
    assert check.relative_error < 0.0
    assert len(check.caveats) >= 4


def test_the_equipotency_prediction_barely_depends_on_moiety_composition():
    """The composition of the moiety is unknown, so the check must not need it.

    Risperidone and paliperidone differ by 4% in molecular weight, so under
    molar equipotency the predicted mass-basis constant spans only that much
    across every possible composition. A comparison that swung with the
    unknown would not be a comparison.
    """
    values = [equipotency_check(f).predicted_ng_per_mL for f in (0.0, 0.5, 1.0)]
    assert max(values) / min(values) < 1.05


# ---------------------------------------------------------------------------
# Schema: the registry cannot hold a record that decides nothing
# ---------------------------------------------------------------------------


def _valid_kwargs(**overrides):
    base = dict(
        parent="p",
        metabolite="m",
        target="D2",
        disposition="refused",
        reason="because",
        exposure_ratio=0.5,
        exposure_basis="AUC",
        exposure_spread="n = 1",
        exposure_regimen="oral",
        exposure_citation="somewhere",
        exposure_provenance="primary",
        affinity_ratio=2.0,
        affinity_species="human",
        affinity_citation="somewhere",
        ec50_axes_available=("parent",),
    )
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(disposition="allowed"), "is not one of"),
        (dict(reason="   "), "empty reason"),
        (dict(exposure_provenance="hearsay"), "is not one of"),
        (dict(ec50_axes_available=("plasma",)), "is not one of"),
        (dict(affinity_species=None), "without a species"),
        (dict(affinity_ratio=None), "without a ratio"),
        (dict(exposure_ratio=-1.0), "not positive"),
        (dict(affinity_ratio=0.0), "not positive"),
    ],
)
def test_the_record_refuses_to_be_built_incoherently(overrides, match):
    with pytest.raises(ValueError, match=match):
        ActiveMetabolite(**_valid_kwargs(**overrides))


def test_a_bounded_disposition_requires_the_numbers_the_bound_needs():
    with pytest.raises(ValueError, match="wearing the wrong label"):
        ActiveMetabolite(
            **_valid_kwargs(
                disposition="bounded",
                exposure_ratio=None,
                exposure_basis="not published",
            )
        )


def test_the_registry_is_not_an_empty_container():
    """Rule 10: a gate with nothing in it is worse than no gate.

    Every registered record has to carry at least one sourced number, and every
    refusal has to name the source that forces it.
    """
    assert set(registered_parent_drugs()) == {
        "aripiprazole",
        "haloperidol",
        "risperidone",
    }
    for drug in registered_parent_drugs():
        for record in active_metabolites_for(drug):
            assert record.quotes, drug
            assert record.exposure_citation.strip(), drug
            carries_a_number = (
                record.exposure_ratio is not None
                or record.affinity_ratio is not None
                or record.half_life_pair_h is not None
            )
            assert carries_a_number, drug


def test_the_refused_pairs_are_refused_for_different_reasons():
    """The two refusals are not one rule applied twice.

    Aripiprazole has parent-basis constants and fails on regimen and on an
    unsettled exposure ratio; risperidone has no parent-basis constant at all
    and could not use one if it had it, because the parent:metabolite split is
    phenotype-dependent. Collapsing them would lose the fact that a
    dehydro-aripiprazole EC50 would reopen aripiprazole and nothing reopens
    risperidone.
    """
    assert "parent" in ARIPIPRAZOLE_DEHYDRO_ARIPIPRAZOLE.ec50_axes_available
    assert "parent" not in RISPERIDONE_9_HYDROXYRISPERIDONE.ec50_axes_available
    assert RISPERIDONE_9_HYDROXYRISPERIDONE.ec50_axes_available == ("moiety",)
    # Neither has the metabolite axis, which is the one the summed form needs.
    for record in (
        ARIPIPRAZOLE_DEHYDRO_ARIPIPRAZOLE,
        RISPERIDONE_9_HYDROXYRISPERIDONE,
    ):
        assert "metabolite" not in record.ec50_axes_available


def test_the_half_life_pairs_show_why_each_ratio_is_regimen_dependent():
    """The evidence for the aripiprazole refusal is on the record, not in prose."""
    assert ARIPIPRAZOLE_DEHYDRO_ARIPIPRAZOLE.half_life_ratio() > 1.0
    # Risperidone's extensive-metaboliser pair differs sevenfold, which is the
    # phenotype dependence the refusal rests on.
    assert RISPERIDONE_9_HYDROXYRISPERIDONE.half_life_ratio() == pytest.approx(7.0)
    assert HALOPERIDOL_REDUCED_HALOPERIDOL.half_life_ratio() is None


def test_an_asserted_equipotency_is_not_stored_as_a_measured_ratio():
    """The ABILIFY label claims similar affinity and prints no metabolite Ki.

    Storing that claim in ``affinity_ratio`` would turn a sentence into a
    parameter, and the drive ratio would then be computable from a number nobody
    measured. Risperidone's label makes the same unnumbered claim, but its ratio
    does not come from the label: see the next test.
    """
    record = ARIPIPRAZOLE_DEHYDRO_ARIPIPRAZOLE
    assert record.affinity_ratio is None
    assert any("similar" in quote for quote in record.quotes)
    with pytest.raises(ValueError, match="no drive ratio"):
        record.drive_ratio()


def test_risperidones_affinity_ratio_is_measured_and_not_the_labels_assertion():
    """A ratio of two Ki values from one assay is not a Ki used as an EC50.

    Rule 1 forbids substituting an in-vitro constant for an in-vivo one. It does
    not forbid a *ratio* of two same-assay constants, which is what sets the
    relative occupancy drive of two ligands at the same receptor; the haloperidol
    record already rests on one. The distinction the test protects is that the
    number came from Schotte's table and not from the RISPERDAL label, which
    asserts similar activity and prints no number to check it against.
    """
    record = RISPERIDONE_9_HYDROXYRISPERIDONE
    assert record.affinity_species == "human"
    assert not record.affinity_is_cross_species
    assert record.affinity_ratio == pytest.approx(4.8 / 5.9)
    assert "Schotte" in record.affinity_citation
    assert "5.9" in record.affinity_citation and "4.8" in record.affinity_citation
