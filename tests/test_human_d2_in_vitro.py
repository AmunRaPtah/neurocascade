import pytest

from neurocascade.human.d2_ec50 import HALOPERIDOL_D2_PLASMA_EC50_nM
from neurocascade.human.d2_in_vitro import (
    EFFICACY_SCALE,
    EXCLUDED_SOURCES,
    HALOPERIDOL_HUMAN_D2_BINDING,
    HALOPERIDOL_HUMAN_D2_FUNCTIONAL,
    HALOPERIDOL_HUMAN_D2_INTRINSIC_EFFICACY,
    INVERSE_AGONISM_ABSTRACT_ONLY_SUPPORT,
    REGISTRY_EFFICACY_LABEL,
    REGISTRY_KI_LABEL,
    SEARCHED_AND_EMPTY,
    binding_ki_fold_spread,
    binding_ki_range_nM,
    efficacy_range,
    haloperidol_d2_binding,
    in_vitro_in_vivo_ratio_range,
    in_vitro_in_vivo_ratios,
    ratio_straddles_unity,
    registry_efficacy,
    registry_ki,
)


def test_seven_primary_human_binding_constants_are_registered():
    assert len(HALOPERIDOL_HUMAN_D2_BINDING) == 7
    assert len({r.label for r in HALOPERIDOL_HUMAN_D2_BINDING}) == 7
    values = [r.value for r in HALOPERIDOL_HUMAN_D2_BINDING]
    assert values == sorted(values), "records are ordered lowest Ki first"


def test_the_spread_between_them_is_the_headline_result():
    """One number would have hidden this. Sixty-eight-fold, all human D2, all primary."""
    low, high = binding_ki_range_nM()
    assert (low, high) == (0.13, 8.85)
    assert binding_ki_fold_spread() == pytest.approx(68.1, abs=0.2)


def test_every_binding_record_carries_the_conditions_needed_to_read_it():
    for record in HALOPERIDOL_HUMAN_D2_BINDING:
        assert record.is_binding_constant
        assert record.ligand == "haloperidol"
        assert record.isoform in {"D2S", "D2L"}
        assert record.tracer and record.tracer_class
        assert record.citation and record.pmid and record.access_route
        assert record.printed_or_converted
        # Whatever a source failed to print is recorded rather than left blank,
        # so a later reader cannot mistake an unchecked field for a checked one.
        for field in (record.sodium, record.temperature, record.incubation_time):
            assert field, "%s left a condition field empty instead of saying so" % record.label


def test_the_only_back_converted_value_says_so():
    converted = [
        r for r in HALOPERIDOL_HUMAN_D2_BINDING if "converted" in r.printed_or_converted
    ]
    assert [r.label for r in converted] == ["sykes2017"]
    assert "logs only" in converted[0].printed_or_converted


def test_registry_ki_is_the_best_documented_source_not_the_most_flattering():
    record = registry_ki()
    assert record.label == REGISTRY_KI_LABEL == "bartuzi2022"
    assert record.value == 8.85
    # It is the highest of the seven, so choosing it cannot be an attempt to make
    # the in-vitro and in-vivo constants look closer than they are.
    assert record.value == max(r.value for r in HALOPERIDOL_HUMAN_D2_BINDING)
    # It is chosen because it prints all of these together; no other source does.
    assert "1.5 nM" in record.tracer
    assert "120 mM NaCl" in record.sodium
    assert record.temperature == "25 C"
    assert record.incubation_time == "120 min"
    assert "sulpiride" in record.nonspecific_binding


def test_functional_constants_are_kept_out_of_the_binding_set():
    binding_labels = {r.label for r in HALOPERIDOL_HUMAN_D2_BINDING}
    for record in HALOPERIDOL_HUMAN_D2_FUNCTIONAL:
        assert record.value_kind == "Kb"
        assert not record.is_binding_constant
        assert record.label not in binding_labels


def test_within_paper_binding_versus_functional_divergence_is_recorded():
    """Same lab, same paper, twenty to sixty-fold apart. Not a transcription error."""
    binding = {r.label: r.value for r in HALOPERIDOL_HUMAN_D2_BINDING}
    functional = {r.label: r.value for r in HALOPERIDOL_HUMAN_D2_FUNCTIONAL}
    assert binding["kaczor2023a"] / functional["kaczor2023a-cAMP"] > 50
    assert binding["kaczor2023b"] / functional["kaczor2023b-cAMP"] > 20


def test_haloperidol_is_an_inverse_agonist_by_two_readouts():
    assert len(HALOPERIDOL_HUMAN_D2_INTRINSIC_EFFICACY) == 2
    readouts = {r.readout for r in HALOPERIDOL_HUMAN_D2_INTRINSIC_EFFICACY}
    assert len(readouts) == 2
    for record in HALOPERIDOL_HUMAN_D2_INTRINSIC_EFFICACY:
        assert record.efficacy < 0, "a neutral antagonist would be 0"
        assert "inverse agonist" in record.classification
        assert "butaclamol" in record.reference_compound
        assert record.constitutive_activity
    low, high = efficacy_range()
    assert (low, high) == (-0.95, -0.83)


def test_efficacy_records_name_their_own_limitations():
    for record in HALOPERIDOL_HUMAN_D2_INTRINSIC_EFFICACY:
        assert record.caveats
        joined = " ".join(record.caveats)
        # The two quantitative sources share a laboratory, which is a real limit
        # on how independent the agreement between them is.
        assert "laboratory" in joined
        assert "EFFICACY_SCALE" in joined
    gtp = next(
        r for r in HALOPERIDOL_HUMAN_D2_INTRINSIC_EFFICACY if r.label == "roberts2005-GTPgS"
    )
    # Its 83% was measured under conditions chosen to maximise the signal.
    assert any("condition-specific" in caveat for caveat in gtp.caveats)


def test_the_sign_convention_is_written_down_and_asymmetric():
    assert "neutral antagonist" in EFFICACY_SCALE
    assert "butaclamol" in EFFICACY_SCALE
    assert "not interconvertible" in EFFICACY_SCALE


def test_registry_efficacy_is_the_whole_cell_readout():
    record = registry_efficacy()
    assert record.label == REGISTRY_EFFICACY_LABEL == "hall1997-cAMP"
    assert record.efficacy == -0.95
    assert "whole cells" in record.readout


def test_ratios_are_reported_as_a_set_because_one_number_would_be_a_choice():
    ratios = dict(in_vitro_in_vivo_ratios())
    assert set(ratios) == {r.label for r in HALOPERIDOL_HUMAN_D2_BINDING}
    for label, value in ratios.items():
        expected = HALOPERIDOL_D2_PLASMA_EC50_nM / dict(
            (r.label, r.value) for r in HALOPERIDOL_HUMAN_D2_BINDING
        )[label]
        assert value == pytest.approx(expected)
    assert ratios["roberts2005"] == pytest.approx(14.50, abs=0.02)
    assert ratios["bartuzi2022"] == pytest.approx(0.213, abs=0.002)


def test_the_ratio_set_straddles_unity():
    """The argument against ever substituting a Ki for the in-vivo EC50.

    The in-vitro sources do not agree on whether the in-vivo constant is larger
    or smaller than they are, so there is no correction factor that could be
    applied to a Ki to obtain an EC50 in the first place.
    """
    assert ratio_straddles_unity()
    low, high = in_vitro_in_vivo_ratio_range()
    assert low < 1.0 < high
    assert low == pytest.approx(0.213, abs=0.002)
    assert high == pytest.approx(14.50, abs=0.02)


def test_ratios_track_the_ec50_they_are_given():
    doubled = dict(in_vitro_in_vivo_ratios(ec50_nM=2 * HALOPERIDOL_D2_PLASMA_EC50_nM))
    base = dict(in_vitro_in_vivo_ratios())
    for label in base:
        assert doubled[label] == pytest.approx(2 * base[label])


def test_excluded_sources_each_carry_a_stated_reason():
    assert len(EXCLUDED_SOURCES) >= 5
    reasons = dict(EXCLUDED_SOURCES)
    assert all(len(reason) > 40 for reason in reasons.values())
    # The two exclusions most likely to be reversed by a careless later reader.
    de_witte = next(v for k, v in reasons.items() if "de Witte" in k)
    assert "does not state" in de_witte and "human" in de_witte
    mccormick = next(v for k, v in reasons.items() if "McCormick" in k)
    assert "Rat brain" in mccormick


def test_paywalled_sources_contribute_no_numbers():
    """Pirate mirrors are out of scope, so paywalled means unavailable, not looked up."""
    reasons = dict(EXCLUDED_SOURCES)
    kroeze = next(v for k, v in reasons.items() if "Kroeze" in k)
    assert "no number taken" in kroeze
    seeman = next(v for k, v in reasons.items() if "Seeman" in k)
    assert "no number from it used" in seeman
    # Burstein is paywalled too, so it supports the direction without a value.
    assert "abstract" in INVERSE_AGONISM_ABSTRACT_ONLY_SUPPORT
    assert "no efficacy value is printed" in INVERSE_AGONISM_ABSTRACT_ONLY_SUPPORT
    assert not any(
        r.citation.startswith("Burstein") for r in HALOPERIDOL_HUMAN_D2_INTRINSIC_EFFICACY
    )


def test_empty_searches_are_recorded_as_results():
    topics = dict(SEARCHED_AND_EMPTY)
    assert "human tissue D2 affinity" in topics
    # No human tissue value was found, so every registered value is from a cell
    # line, and that has to be true of the records as well as of the note.
    assert "heterologous cell line" in topics["human tissue D2 affinity"]
    for record in HALOPERIDOL_HUMAN_D2_BINDING:
        assert "CHO" in record.preparation or "HEK293" in record.preparation
    arrestin = topics["beta-arrestin intrinsic efficacy at human D2"]
    assert "Only the non-retrieval is claimed here." in arrestin


def test_transcription_traps_are_written_next_to_the_records_they_threaten():
    by_label = {r.label: r for r in HALOPERIDOL_HUMAN_D2_BINDING}
    # Both Kaczor papers print D1 and D3 haloperidol values in an adjacent table.
    assert "D1 and D3" in by_label["kaczor2023a"].notes
    assert "D1 and D3" in by_label["kaczor2023b"].notes
    # The two BJP PDFs render the micro sign as "m" under pdftotext.
    for label in ("roberts2005", "hall1997"):
        assert "corrected reading" in by_label[label].notes
    # roberts2005 prints three constants; the record says which one it took.
    assert "three constants" in by_label["roberts2005"].notes


def test_lookup_by_label_and_the_registry_labels_resolve():
    assert haloperidol_d2_binding("bartuzi2022")[0].value == 8.85
    assert haloperidol_d2_binding("not-a-source") == ()
    assert haloperidol_d2_binding() is HALOPERIDOL_HUMAN_D2_BINDING
