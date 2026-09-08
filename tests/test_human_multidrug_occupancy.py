"""P21.6: the dose-to-occupancy chain is keyed by drug, not fixed to haloperidol.

The chain runs for any drug that has both a complete human PK set and a
registered in-vivo D2 EC50, and refuses per-drug -- naming the layer that is
missing -- for a drug that lacks either. Today haloperidol is the only drug with
both in the registries, so the refusals are exercised against the real registry
gaps and the "runs for another drug" case is exercised by relabelling
haloperidol's own sourced records onto a second drug key, which keeps every
number a sourced one while proving the key threads all the way through.
"""
from dataclasses import replace

import numpy as np
import pytest

from neurocascade.human.dose_to_occupancy import predict_occupancy
from neurocascade.human.occupancy_validation import (
    chain_testable_observations,
    dose_reaching_occupancy,
    steady_state_occupancy,
)
from neurocascade.human.pk_parameters import human_pk_records
from neurocascade.human.receptor_parameters import require_human_parameter


def _relabelled_records(drug):
    """Haloperidol's own PK and receptor records, relabelled onto ``drug``.

    Relabelling rather than inventing: the point is that the drug key resolves
    and threads through, so every parameter stays a sourced haloperidol one and
    nothing but the key changes.
    """
    pk = replace(human_pk_records("haloperidol", "PO")[0], drug=drug)
    receptor = require_human_parameter("D2")
    receptor = replace(
        receptor,
        ligand=drug,
        ec50_record=replace(receptor.ec50_record, drug=drug),
    )
    return pk, receptor


def test_haloperidol_is_still_the_default_and_is_unchanged():
    default = predict_occupancy(5.0)
    named = predict_occupancy(5.0, drug="haloperidol")
    assert default.drug == "haloperidol"
    assert named.drug == "haloperidol"
    assert np.array_equal(default.occupancy, named.occupancy)
    assert default.ec50_nM == named.ec50_nM


def test_a_drug_with_neither_pk_nor_ec50_is_refused_at_the_pk_layer():
    # PK is resolved before the receptor, so a drug the registries do not know
    # at all is refused for the first missing layer, which is the PK set.
    with pytest.raises(ValueError, match="no human clozapine PK record"):
        predict_occupancy(5.0, drug="clozapine")


def test_a_drug_with_pk_but_no_ec50_is_refused_at_the_occupancy_layer():
    # Give the drug a complete PK set (haloperidol's, relabelled) but leave the
    # EC50 registry with no row for it: the refusal must now come from the
    # occupancy layer and name the drug, not fall back to haloperidol's EC50.
    pk, _ = _relabelled_records("clozapine")
    with pytest.raises(ValueError, match="clozapine") as excinfo:
        predict_occupancy(5.0, drug="clozapine", pk_record=pk)
    assert "p21-2" in str(excinfo.value).lower() or "EC50" in str(excinfo.value)


def test_a_drug_with_ec50_but_no_pk_is_refused_at_the_pk_layer():
    # The mirror case: a receptor record is supplied but no PK set exists, so the
    # PK layer is where it stops, naming the drug.
    _, receptor = _relabelled_records("clozapine")
    with pytest.raises(ValueError, match="no human clozapine PK record"):
        predict_occupancy(5.0, drug="clozapine", receptor_record=receptor)


def test_the_chain_runs_for_another_drug_when_all_layers_are_supplied():
    # Fentanyl is the relabel target because it carries a Kp,uu record whose
    # value is absent, so the no-brain-partition invariant holds for it the way
    # it does for haloperidol; a drug with no Kp,uu record at all is refused at
    # the exposure layer (that gate is exercised elsewhere). The PK set and EC50
    # are supplied as haloperidol's own, relabelled.
    pk, receptor = _relabelled_records("fentanyl")
    prediction = predict_occupancy(
        5.0, drug="fentanyl", pk_record=pk, receptor_record=receptor
    )
    assert prediction.drug == "fentanyl"
    assert prediction.target == "D2"
    # Relabelled haloperidol records, so the numbers must match the haloperidol
    # run exactly: the key changed, the sourced values did not.
    haloperidol = predict_occupancy(5.0)
    assert np.array_equal(prediction.occupancy, haloperidol.occupancy)
    assert prediction.ec50_nM == haloperidol.ec50_nM


def test_a_pk_record_naming_a_different_drug_than_the_key_is_refused():
    pk = replace(human_pk_records("haloperidol", "PO")[0], drug="risperidone")
    with pytest.raises(ValueError, match="human clozapine only"):
        predict_occupancy(5.0, drug="clozapine", pk_record=pk)


def test_steady_state_and_dose_reaching_forward_the_drug_key():
    # A drug the registries do not know is refused the same way through the
    # regimen wrappers, which proves the key is forwarded rather than dropped.
    with pytest.raises(ValueError, match="no human clozapine PK record"):
        steady_state_occupancy(2.0, drug="clozapine")
    with pytest.raises(ValueError, match="no human clozapine PK record"):
        dose_reaching_occupancy(0.7, drug="clozapine")

    pk, receptor = _relabelled_records("fentanyl")
    state = steady_state_occupancy(
        2.0, n_doses=8, drug="fentanyl", pk_record=pk, receptor_record=receptor
    )
    reference = steady_state_occupancy(2.0, n_doses=8)
    assert state.mean_occupancy == pytest.approx(reference.mean_occupancy)


def test_chain_testable_observations_is_drug_keyed():
    # Haloperidol: the P19.6 empty tuple is preserved for the default and the
    # explicit key alike.
    assert chain_testable_observations() == ()
    assert chain_testable_observations("haloperidol") == ()
    # Another drug: refused, not silently empty, because "no observations
    # registered" is a different claim from "observations exist but none is
    # chain-compatible".
    with pytest.raises(ValueError, match="no registered human D2 PET observations"):
        chain_testable_observations("risperidone")
