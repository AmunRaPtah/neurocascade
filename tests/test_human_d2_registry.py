"""P21.1: the drug-keyed D2 plasma EC50 registry.

Two things are being tested here. The first is that the refactor moved nothing:
the haloperidol arm of the chain must predict bit-identically to the code that
existed before the registry, and `tests/fixtures/p21_1_haloperidol_pre_refactor.
json` was recorded from that code at commit b6b0edc for exactly this purpose.
The comparison is `==` on sha256 digests of the raw float64 bytes and on C99
hex-encoded scalars, not `approx`, because "bit-identical" is the acceptance
criterion and a tolerance would not test it.

The second is that the schema enforces what P21.2 found the sources actually
print. Those checks construct throwaway records; none of them adds a drug to the
registry, which still has exactly one row.
"""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from neurocascade.human.d2_ec50 import (
    D2_PLASMA_EC50,
    D2PlasmaEC50,
    EC50_PROVENANCE_CLASSES,
    EMAX_CONVENTIONS,
    HALOPERIDOL_D2_PLASMA_EC50,
    HALOPERIDOL_D2_PLASMA_EC50_nM,
    HALOPERIDOL_PLASMA_FREE_FRACTION,
    PLASMA_BASES,
    compare_with_reported_relations,
    d2_plasma_ec50,
    ec50_landscape,
    fit_plasma_ec50,
    registered_d2_drugs,
)
from neurocascade.human.dose_to_occupancy import predict_occupancy
from neurocascade.human.pet_validation import HALOPERIDOL_MW_G_PER_MOL
from neurocascade.human.receptor_parameters import require_human_parameter

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "p21_1_haloperidol_pre_refactor.json"


@pytest.fixture(scope="module")
def golden():
    with GOLDEN_PATH.open() as handle:
        return json.load(handle)


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=float).tobytes()).hexdigest()


# --------------------------------------------------------------------------
# Nothing moved
# --------------------------------------------------------------------------


def test_the_recorded_constants_are_the_same_floats_as_before_the_refactor(golden):
    assert (
        float(HALOPERIDOL_D2_PLASMA_EC50_nM).hex()
        == golden["recorded_constant_nM_hex"]
    )
    assert (
        float(HALOPERIDOL_PLASMA_FREE_FRACTION).hex() == golden["free_fraction_hex"]
    )
    # Derived from the row rather than duplicated beside it, so the two names
    # cannot drift apart.
    assert HALOPERIDOL_D2_PLASMA_EC50_nM is HALOPERIDOL_D2_PLASMA_EC50.ec50_nM
    assert (
        HALOPERIDOL_PLASMA_FREE_FRACTION
        is HALOPERIDOL_D2_PLASMA_EC50.plasma_free_fraction
    )


def test_the_fit_and_everything_derived_from_it_are_unchanged(golden):
    fit = fit_plasma_ec50()
    free = fit_plasma_ec50(plasma_basis="free")
    assert float(fit.ec50_nM).hex() == golden["refit_ec50_nM_hex"]
    assert float(free.ec50_nM).hex() == golden["refit_free_basis_ec50_nM_hex"]
    assert float(fit.interval_low_nM).hex() == golden["refit_interval_low_nM_hex"]
    assert float(fit.interval_high_nM).hex() == golden["refit_interval_high_nM_hex"]
    assert (
        float(fit.standard_error_nM).hex() == golden["refit_standard_error_nM_hex"]
    )
    assert [[label, float(value).hex()] for label, value in ec50_landscape_values()] == [
        [label, value] for label, value in golden["landscape"]
    ]
    assert [
        [c.label, float(c.ratio_fitted_over_reported).hex()]
        for c in compare_with_reported_relations()
    ] == [[label, value] for label, value in golden["reported_comparisons"]]


def ec50_landscape_values():
    return [(row[0], row[1]) for row in ec50_landscape()]


@pytest.mark.parametrize("index", range(4))
def test_haloperidol_predictions_are_bit_identical_to_the_pre_refactor_chain(
    golden, index
):
    """The acceptance criterion of P21.1, as an exact comparison.

    A refactor that changed a predicted occupancy by one ulp would still be a
    refactor that changed the model, so this is equality on the bytes.
    """
    expected = golden["predictions"][index]
    prediction = predict_occupancy(expected["dose_mg"])

    assert float(prediction.ec50_nM).hex() == expected["ec50_nM_hex"]
    assert _digest(prediction.times_min) == expected["times_min_sha256"]
    assert _digest(prediction.plasma_nM) == expected["plasma_nM_sha256"]
    assert _digest(prediction.occupancy) == expected["occupancy_sha256"]
    assert float(prediction.peak_plasma_nM).hex() == expected["peak_plasma_nM_hex"]
    assert float(prediction.peak_occupancy).hex() == expected["peak_occupancy_hex"]
    assert (
        float(prediction.time_of_peak_occupancy_min).hex()
        == expected["time_of_peak_occupancy_min_hex"]
    )
    assert (
        float(prediction.occupancy_at(240.0)).hex()
        == expected["occupancy_at_240min_hex"]
    )


def test_the_registry_row_reproduces_the_chains_own_hyperbola():
    """The row's `occupancy_at` is the chain's model, not a second one."""
    prediction = predict_occupancy(5.0)
    row = HALOPERIDOL_D2_PLASMA_EC50
    for concentration, occupancy in zip(prediction.plasma_nM, prediction.occupancy):
        assert occupancy == row.occupancy_at(float(concentration))


def test_the_receptor_registry_now_points_at_the_row_it_copied_from():
    d2 = require_human_parameter("D2")
    assert d2.ec50_record is HALOPERIDOL_D2_PLASMA_EC50
    assert d2.ec50_in_vivo_nM == HALOPERIDOL_D2_PLASMA_EC50.ec50_nM
    assert d2.ec50_plasma_basis == HALOPERIDOL_D2_PLASMA_EC50.plasma_basis
    assert d2.ec50_origin == HALOPERIDOL_D2_PLASMA_EC50.provenance


# --------------------------------------------------------------------------
# The registry is one row, and says so
# --------------------------------------------------------------------------


def test_the_registry_holds_exactly_the_one_drug_p21_1_was_allowed_to_add():
    """P21.1 is a pure refactor. A second row here means a drug arrived without
    a source, which is the failure mode the whole phase is ordered to avoid."""
    assert registered_d2_drugs() == ("haloperidol",)
    assert set(D2_PLASMA_EC50) == {"haloperidol"}
    assert d2_plasma_ec50("haloperidol") is HALOPERIDOL_D2_PLASMA_EC50


def test_an_unregistered_drug_is_a_named_refusal_not_a_key_error():
    with pytest.raises(ValueError) as error:
        d2_plasma_ec50("clozapine")
    message = str(error.value)
    assert "haloperidol" in message
    # The message points at the reachability record rather than implying the
    # value could simply be typed in.
    assert "p21-2" in message


def test_the_registry_cannot_be_mutated_at_runtime():
    with pytest.raises(TypeError):
        D2_PLASMA_EC50["clozapine"] = HALOPERIDOL_D2_PLASMA_EC50


def test_haloperidols_row_records_what_its_sources_did_not_print():
    row = HALOPERIDOL_D2_PLASMA_EC50
    # P19.7: all four fitted points carry region=None and tracer=None. The row
    # says so rather than inheriting "striatum" from the surrounding prose.
    assert row.region is None
    assert row.tracer is None
    assert row.provenance == "fitted-in-repo"
    assert row.emax_convention == "fixed-at-1" and row.emax == 1.0
    assert row.plasma_basis == "total" and row.analyte_is_summed is False
    assert "depot" in row.regimen and "intramuscular" in row.regimen
    assert row.molar_mass_g_per_mol == HALOPERIDOL_MW_G_PER_MOL


def test_unit_conversion_lands_on_the_axis_the_literature_prints():
    row = HALOPERIDOL_D2_PLASMA_EC50
    assert row.ec50_ng_per_mL == pytest.approx(
        row.ec50_nM * HALOPERIDOL_MW_G_PER_MOL / 1000.0
    )
    assert row.ec50_ng_per_mL == pytest.approx(0.708, abs=0.005)


def test_the_interval_travels_with_the_constant_and_contains_it():
    row = HALOPERIDOL_D2_PLASMA_EC50
    assert row.interval_low_nM < row.ec50_nM < row.interval_high_nM
    assert "not independent" in row.interval_caveat
    assert "week 1 and week 4" in row.interval_caveat
    assert row.interval_method


# --------------------------------------------------------------------------
# The schema enforces P21.2's three findings
# --------------------------------------------------------------------------


def test_an_ec50_cannot_be_registered_without_the_emax_it_was_fitted_against():
    """P21.2's first schema requirement. Clozapine moves 105 to 483 ng/mL on
    this choice alone, so a row that omits it stores an ambiguous number."""
    with pytest.raises(ValueError):
        replace(HALOPERIDOL_D2_PLASMA_EC50, emax_convention="whatever-was-handy")
    # An Emax of 0.6 is a real fitted value, but not under a convention that
    # claims the fit pinned it at 1.
    with pytest.raises(ValueError) as error:
        replace(HALOPERIDOL_D2_PLASMA_EC50, emax=0.6)
    assert "fixed-at-1" in str(error.value)
    # Declared as fitted-free, 0.6 is accepted, and it changes the prediction.
    free_emax = replace(
        HALOPERIDOL_D2_PLASMA_EC50, emax=0.6, emax_convention="fitted-free"
    )
    assert free_emax.occupancy_at(free_emax.ec50_nM) == pytest.approx(0.3)


def test_conventions_and_classes_are_closed_vocabularies():
    assert set(EMAX_CONVENTIONS) == {"fixed-at-1", "fitted-free"}
    assert set(PLASMA_BASES) == {"total", "free"}
    assert "second-hand" in EC50_PROVENANCE_CLASSES
    with pytest.raises(ValueError):
        replace(HALOPERIDOL_D2_PLASMA_EC50, plasma_basis="unbound")
    with pytest.raises(ValueError):
        replace(HALOPERIDOL_D2_PLASMA_EC50, provenance="probably-fine")


def test_a_second_hand_value_must_name_where_it_was_actually_read():
    """P21.2 found a review pairing an Emax and an EC50 that never went
    together in the source. A row read out of a quoting paper has to say so."""
    with pytest.raises(ValueError) as error:
        replace(HALOPERIDOL_D2_PLASMA_EC50, provenance="second-hand", read_in=None)
    assert "read_in" in str(error.value)
    replace(
        HALOPERIDOL_D2_PLASMA_EC50,
        provenance="second-hand",
        read_in="a paper that quotes the primary",
    )


def test_a_free_fraction_needs_its_source_and_gates_the_basis_conversion():
    row = HALOPERIDOL_D2_PLASMA_EC50
    assert row.ec50_on_basis("total") == row.ec50_nM
    assert row.ec50_on_basis("free") == pytest.approx(
        row.ec50_nM * HALOPERIDOL_PLASMA_FREE_FRACTION
    )
    with pytest.raises(ValueError):
        replace(row, free_fraction_citation=None)
    # Without a free fraction the conversion is refused rather than assumed.
    bare = replace(row, plasma_free_fraction=None, free_fraction_citation=None)
    with pytest.raises(ValueError) as error:
        bare.ec50_on_basis("free")
    assert "no plasma free fraction is recorded" in str(error.value)


def test_the_interval_must_bracket_the_value_it_belongs_to():
    with pytest.raises(ValueError):
        replace(HALOPERIDOL_D2_PLASMA_EC50, interval_low_nM=2.0)
    with pytest.raises(ValueError):
        replace(HALOPERIDOL_D2_PLASMA_EC50, interval_high_nM=None)
    with pytest.raises(ValueError):
        replace(HALOPERIDOL_D2_PLASMA_EC50, interval_method="")


def test_comparability_names_every_axis_two_constants_differ_on():
    """The schema's job is to make a cross-drug comparison state its terms.

    The second record here is a construction, not a registry entry: it carries
    the shape of the risperidone row P21.2 describes without asserting any of
    its numbers, which is why its EC50 is haloperidol's.
    """
    row = HALOPERIDOL_D2_PLASMA_EC50
    assert row.comparability_conflicts(row) == ()
    row.require_comparable_with(row)

    other = replace(
        row,
        drug="a-drug-shaped-hole",
        analyte="parent plus its active metabolite, summed",
        analyte_is_summed=True,
        emax=0.88,
        emax_convention="fitted-free",
        region="putamen",
        tracer="[11C]raclopride",
        provenance="second-hand",
        read_in="a source quoting the primary",
    )
    conflicts = other.comparability_conflicts(row)
    joined = "; ".join(conflicts)
    assert "analyte" in joined
    assert "Emax convention" in joined
    assert "region" in joined
    assert "tracer" in joined
    with pytest.raises(ValueError) as error:
        other.require_comparable_with(row)
    assert "not the same measured quantity" in str(error.value)


def test_the_chain_refuses_a_summed_moiety_constant():
    """The PK layer integrates the parent drug alone, so a summed-moiety EC50
    is the wrong quantity for it. Refused, because the error is silent."""
    d2 = require_human_parameter("D2")
    summed = replace(
        d2,
        ec50_record=replace(
            HALOPERIDOL_D2_PLASMA_EC50,
            analyte="parent plus its active metabolite, summed",
            analyte_is_summed=True,
        ),
    )
    with pytest.raises(ValueError) as error:
        predict_occupancy(5.0, receptor_record=summed)
    assert "summed active moiety" in str(error.value)


def test_the_chain_reports_the_emax_convention_in_its_occupancy_layer():
    layer = predict_occupancy(5.0).layer("occupancy")
    assert "fixed-at-1" in layer.detail
    assert "Emax" in layer.detail


def test_a_row_must_be_a_positive_finite_number_with_a_source():
    with pytest.raises(ValueError):
        replace(HALOPERIDOL_D2_PLASMA_EC50, ec50_nM=0.0)
    with pytest.raises(ValueError):
        replace(HALOPERIDOL_D2_PLASMA_EC50, ec50_nM=float("nan"))
    with pytest.raises(ValueError):
        replace(HALOPERIDOL_D2_PLASMA_EC50, citation="")
    with pytest.raises(ValueError):
        replace(HALOPERIDOL_D2_PLASMA_EC50, analyte="")
    with pytest.raises(ValueError):
        replace(HALOPERIDOL_D2_PLASMA_EC50, regimen="")
    with pytest.raises(ValueError):
        replace(HALOPERIDOL_D2_PLASMA_EC50, molar_mass_g_per_mol=0.0)
    with pytest.raises(ValueError):
        D2PlasmaEC50(
            drug="",
            ec50_nM=1.0,
            molar_mass_g_per_mol=1.0,
            plasma_basis="total",
            analyte="parent",
            analyte_is_summed=False,
            emax=1.0,
            emax_convention="fixed-at-1",
            region=None,
            tracer=None,
            regimen="oral",
            provenance="primary",
            citation="somewhere",
        )


def test_occupancy_at_rejects_negative_concentrations():
    with pytest.raises(ValueError):
        HALOPERIDOL_D2_PLASMA_EC50.occupancy_at(-1.0)
    assert HALOPERIDOL_D2_PLASMA_EC50.occupancy_at(0.0) == 0.0


def test_summary_prints_the_qualifiers_rather_than_just_the_number():
    text = HALOPERIDOL_D2_PLASMA_EC50.summary()
    assert "haloperidol" in text
    assert "1.8849 nM" in text
    assert "not printed" in text  # region and tracer
    assert "fixed-at-1" in text
