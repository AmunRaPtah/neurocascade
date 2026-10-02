"""P29.1: the translation-aware species gate.

A non-human number enters a human prediction only under a TranslationTag naming
a registered, cited method; untagged, it is refused as before. The admitted
record propagates into the prediction's provenance.
"""

import math

import numpy as np
import pytest

from neurocascade import species_gate
from neurocascade.human.dose_to_occupancy import ChainLayer, OccupancyPrediction
from neurocascade.human.kp_uu import (
    HUMAN_KP_UU_CSF,
    IDENTITY_TRANSFER_FACTOR,
    score_rat_kp_uu_transfer,
)
from neurocascade.species_gate import (
    TRANSLATION_METHODS,
    TranslatedInput,
    TranslationTag,
    UnsourcedParameterError,
    gate,
)


def _rat_tag(method="kp-uu-transfer"):
    return TranslationTag(method, "rat", "Friden 2009, PMID 19764786")


# --- the gate: both paths -----------------------------------------------------


def test_an_untagged_non_human_number_is_still_refused():
    with pytest.raises(UnsourcedParameterError, match="enter a human prediction"):
        gate("rat", layer="exposure layer", label="Kp,uu")


def test_a_tagged_non_human_number_is_admitted_as_a_translated_input():
    admitted = gate("Rattus norvegicus", layer="exposure layer",
                    label="diazepam Kp,uu", source="CHEMBL1798466",
                    translation=_rat_tag())
    assert isinstance(admitted, TranslatedInput)
    assert admitted.species == "rat"
    assert admitted.method == "kp-uu-transfer"
    assert admitted.layer == "exposure layer"
    assert admitted.label == "diazepam Kp,uu"
    assert admitted.source == "CHEMBL1798466"
    assert "PMID 19764786" in admitted.describe()


def test_reporting_mode_does_not_license_an_untagged_number_either():
    # allow_anchors is the P11.2 reporting mode; a mouse number no anchor covers
    # is refused there too, and only a tag admits it.
    with pytest.raises(UnsourcedParameterError):
        gate("mouse", layer="PK layer", label="CL", allow_anchors=True)
    admitted = gate("mouse", layer="PK layer", label="CL",
                    translation=TranslationTag("allometric-scaling", "mouse", "src"))
    assert admitted.method == "allometric-scaling"


def test_human_and_no_claim_parameters_pass_untagged_as_before():
    assert gate("human", layer="l", label="x") is None
    assert gate(None, layer="l", label="x") is None


def test_a_tag_licenses_only_the_species_it_names():
    with pytest.raises(ValueError, match="licenses only the species it names"):
        gate("mouse", layer="l", label="x", translation=_rat_tag())


def test_a_tag_on_a_human_or_no_claim_parameter_is_an_error():
    for species in ("human", None):
        with pytest.raises(ValueError, match="no species crossing"):
            gate(species, layer="l", label="x", translation=_rat_tag())


def test_the_translation_argument_must_be_a_tag():
    with pytest.raises(TypeError, match="TranslationTag"):
        gate("rat", layer="l", label="x", translation="allometric-scaling")


# --- the tag ------------------------------------------------------------------


def test_the_registered_methods_are_pinned_with_their_citations():
    assert {key: method.citation for key, method in TRANSLATION_METHODS.items()} == {
        "allometric-scaling":
            "Boxenbaum 1982, J Pharmacokinet Biopharm 10:201-227, PMID 7120049",
        "ivive": "Obach 1999, Drug Metab Dispos 27:1350-1359, PMID 10534321",
        "kp-uu-transfer":
            "Friden 2009, J Med Chem 52:6233-6243, PMID 19764786 (rank order "
            "of rat and human Kp,uu,CSF similar across 43 drugs)",
    }


def test_a_tag_requires_a_registered_method():
    with pytest.raises(ValueError, match="not registered"):
        TranslationTag("vibes", "rat", "src")


def test_a_tag_requires_a_recognized_non_human_species():
    with pytest.raises(ValueError, match="FROM a non-human"):
        TranslationTag("ivive", "human", "src")
    with pytest.raises(ValueError, match="FROM a non-human"):
        TranslationTag("ivive", None, "src")
    with pytest.raises(ValueError, match="not a recognized species"):
        TranslationTag("ivive", "HEK293", "src")


def test_a_tag_requires_a_measurement_citation():
    with pytest.raises(ValueError, match="cite the measurement"):
        TranslationTag("ivive", "rat", "  ")


def test_anchors_still_never_drive_a_prediction():
    assert species_gate.CROSS_SPECIES_ANCHORS["morphine/kp-uu"].may_drive_predictions is False


# --- provenance propagation -----------------------------------------------------


def _prediction(layers):
    return OccupancyPrediction(
        drug="diazepam", target="D2", dose_mg=1.0, route="PO", body_weight_kg=70.0,
        times_min=np.array([0.0, 1.0]), plasma_nM=np.zeros(2), occupancy=np.zeros(2),
        concentration_basis="total plasma", receptor_compartment="brain",
        ec50_nM=1.0, layers=layers,
    )


def test_the_prediction_records_which_inputs_were_translated_and_how():
    admitted = gate("rat", layer="exposure layer", label="Kp,uu",
                    source="CHEMBL1798466", translation=_rat_tag())
    prediction = _prediction((
        ChainLayer("pk", "sourced", "d", "c", "measured"),
        ChainLayer("exposure", "translated", "d", "c", "translated",
                   translations=(admitted,)),
    ))
    assert prediction.provenance == "translated"
    assert prediction.translated_inputs == (admitted,)
    assert [(r.label, r.species, r.method) for r in prediction.translated_inputs] == [
        ("Kp,uu", "rat", "kp-uu-transfer")
    ]
    assert "translated input: exposure layer: Kp,uu translated from rat by kp-uu-transfer" \
        in prediction.summary()


def test_an_untranslated_prediction_reports_no_translated_inputs():
    prediction = _prediction((ChainLayer("pk", "sourced", "d", "c", "measured"),))
    assert prediction.translated_inputs == ()
    assert prediction.provenance == "measured"


def test_illustrative_still_outranks_translated():
    admitted = gate("rat", layer="l", label="x", translation=_rat_tag())
    prediction = _prediction((
        ChainLayer("exposure", "t", "d", "c", "translated", translations=(admitted,)),
        ChainLayer("ke0", "illustrative", "d", "c", "illustrative"),
    ))
    assert prediction.provenance == "illustrative"


def test_a_layer_cannot_claim_translated_without_the_gate_record_or_hide_one():
    admitted = gate("rat", layer="l", label="x", translation=_rat_tag())
    with pytest.raises(ValueError, match="go together"):
        ChainLayer("exposure", "t", "d", "c", "translated")
    with pytest.raises(ValueError, match="go together"):
        ChainLayer("exposure", "t", "d", "c", "measured", translations=(admitted,))
    with pytest.raises(TypeError, match="TranslatedInput"):
        ChainLayer("exposure", "t", "d", "c", "translated", translations=("rat",))


# --- the translation scored against human numbers -------------------------------


def test_rat_kp_uu_identity_transfer_against_human_kp_uu_csf():
    """Friden 2009 human Kp,uu,CSF (ChEMBL assay CHEMBL1798468) for the four
    benchmark drugs that have one, against the P28.3 rat Kp,uu,brain carried
    over by the tagged identity transfer."""
    assert IDENTITY_TRANSFER_FACTOR == 1.0
    assert [(r.drug, r.kp_uu_csf) for r in HUMAN_KP_UU_CSF] == [
        ("amitriptyline", 0.18), ("bupropion", 2.0),
        ("diazepam", 0.79), ("thioridazine", 1.4),
    ]
    rows, gmfe = score_rat_kp_uu_transfer()
    got = {r["drug"]: (r["rat_kp_uu_brain"], r["predicted_human_kp_uu"],
                       r["observed_human_kp_uu_csf"]) for r in rows}
    assert got == {
        "amitriptyline": (0.73, 0.73, 0.18),
        "bupropion": (2.0, 2.0, 2.0),
        "diazepam": (1.07, 1.07, 0.79),
        "thioridazine": (0.45, 0.45, 1.4),
    }
    folds = {r["drug"]: r["fold_error"] for r in rows}
    assert folds["amitriptyline"] == pytest.approx(0.73 / 0.18)
    assert folds["bupropion"] == pytest.approx(1.0)
    assert folds["diazepam"] == pytest.approx(1.07 / 0.79)
    assert folds["thioridazine"] == pytest.approx(0.45 / 1.4)
    assert gmfe == pytest.approx(math.exp(sum(abs(math.log(f)) for f in folds.values()) / 4))
    assert gmfe == pytest.approx(2.0332, abs=1e-4)
    within_2x = sum(1 for f in folds.values() if 0.5 <= f <= 2.0)
    assert within_2x == 2
    for row in rows:
        assert row["translation"].method == "kp-uu-transfer"
        assert row["translation"].species == "rat"
        assert "CHEMBL1798466" in row["translation"].source
