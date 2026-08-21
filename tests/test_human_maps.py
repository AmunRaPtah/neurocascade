"""The human receptor-maps join: expression -> whole-brain coupling dict.

Offline, off the committed fixture in `src/neurocascade/human/fixtures/`,
which is a real twelve-region x forty-gene slice of the checksum-verified
ENIGMA AHBA file. The fixture does not include the two all-NaN regions
(`R_frontalpole`, `R_temporalpole`), so the `missing=` policy is exercised
against a small synthetic matrix, exactly as `test_human_expression.py`
does for its own NaN bookkeeping.
"""

import csv

import numpy as np
import pytest

from neurocascade import human
from neurocascade.human import expression as expr
from neurocascade.human import fixtures, maps


@pytest.fixture(scope="module")
def fixture_expression():
    return fixtures.load_fixture_expression()


def _write_csv(tmp_path, header, rows):
    path = tmp_path / "expr.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    return str(path)


def test_whole_brain_nodes_is_identity():
    labels = ["Lcaud", "Lput", "R_insula"]
    assert maps.whole_brain_nodes(labels) == {
        "Lcaud": "Lcaud",
        "Lput": "Lput",
        "R_insula": "R_insula",
    }


def test_unmeasured_regions_is_empty_on_the_fixture(fixture_expression):
    # The twelve fixture regions are all finite by construction; the two
    # all-NaN regions of the full file are not in the slice.
    assert maps.unmeasured_regions(fixture_expression, "DRD2") == []


def test_coupling_keys_follow_expression_region_order(fixture_expression):
    result = maps.whole_brain_coupling(
        fixture_expression, "DRD2", mechanism_gains=-3.0, normalization="max"
    )
    assert list(result.coupling) == list(fixture_expression.region_labels)
    assert result.excluded == ()
    assert result.n_nodes == 12
    assert result.symbol == "DRD2"
    assert result.normalization == "max"


def test_coupling_scalar_gain_equals_relative_density_times_gain(fixture_expression):
    gain = -3.0
    rho = expr.relative_density(
        fixture_expression, "DRD2", normalization="max"
    )
    result = maps.whole_brain_coupling(
        fixture_expression, "DRD2", mechanism_gains=gain, normalization="max"
    )
    for label in fixture_expression.region_labels:
        assert result.coupling[label] == pytest.approx(
            gain * rho.for_region(label)
        )
    # The factorization is `gain * rho` executed, not a second copy of it.
    assert result.coupling["Lcaud"] == pytest.approx(gain * 1.0)


def test_coupling_dict_gains_are_per_node(fixture_expression):
    # A per-node gain dict must cover every node, the same contract
    # `as_coupling` enforces; a subset would silently leave the rest without
    # a drive, so it is refused rather than defaulted to zero.
    gains = {label: -1.0 for label in fixture_expression.region_labels}
    gains["Lcaud"] = -2.0
    gains["Lput"] = -4.0
    rho = expr.relative_density(
        fixture_expression, "DRD2", normalization="max"
    )
    result = maps.whole_brain_coupling(
        fixture_expression, "DRD2", mechanism_gains=gains, normalization="max"
    )
    assert result.coupling["Lcaud"] == pytest.approx(-2.0 * rho.for_region("Lcaud"))
    assert result.coupling["Lput"] == pytest.approx(-4.0 * rho.for_region("Lput"))
    assert result.coupling["Lhippo"] == pytest.approx(-1.0 * rho.for_region("Lhippo"))


def test_coupling_dict_gain_missing_a_node_is_refused(fixture_expression):
    gains = {"Lcaud": -2.0, "Lput": -4.0}
    with pytest.raises(KeyError, match="no mechanism gain"):
        maps.whole_brain_coupling(
            fixture_expression, "DRD2", mechanism_gains=gains, normalization="max"
        )


def test_coupling_drd2_is_striatum_dominant(fixture_expression):
    result = maps.whole_brain_coupling(
        fixture_expression, "DRD2", mechanism_gains=1.0, normalization="max"
    )
    peak = max(result.coupling, key=result.coupling.get)
    assert peak in ("Lcaud", "Lput", "Laccumb"), peak


def test_coupling_htr2a_is_cortex_dominant(fixture_expression):
    result = maps.whole_brain_coupling(
        fixture_expression, "HTR2A", mechanism_gains=1.0, normalization="max"
    )
    order = sorted(result.coupling, key=result.coupling.get, reverse=True)
    top_two = set(order[:2])
    assert top_two <= {
        "L_bankssts", "R_bankssts", "L_precentral", "R_precentral",
        "L_superiorfrontal",
    }, top_two


def test_coupling_passes_normalization_and_reference_through(fixture_expression):
    result = maps.whole_brain_coupling(
        fixture_expression,
        "DRD2",
        mechanism_gains=1.0,
        normalization="reference-region",
        reference="Lput",
    )
    assert result.normalization == "reference-region"
    assert result.reference == "Lput"
    assert result.coupling["Lput"] == pytest.approx(1.0)


def test_coupling_defaults_to_raise_on_an_unmeasured_region(tmp_path):
    path = _write_csv(
        tmp_path,
        ["label", "DRD2"],
        [["Lcaud", "0.5"], ["Rfrontalpole", "NaN"], ["Lput", "0.4"]],
    )
    loaded = expr.read_expression(path, symbols=["DRD2"])
    with pytest.raises(ValueError) as caught:
        maps.whole_brain_coupling(loaded, "DRD2", mechanism_gains=1.0)
    assert "Rfrontalpole" in str(caught.value)
    assert "exclude" in str(caught.value)


def test_coupling_exclude_drops_unmeasured_and_records_it(tmp_path):
    path = _write_csv(
        tmp_path,
        ["label", "DRD2"],
        [["Lcaud", "0.5"], ["Rfrontalpole", "NaN"], ["Lput", "0.4"]],
    )
    loaded = expr.read_expression(path, symbols=["DRD2"])
    result = maps.whole_brain_coupling(
        loaded, "DRD2", mechanism_gains=1.0, missing="exclude"
    )
    assert set(result.coupling) == {"Lcaud", "Lput"}
    assert result.excluded == ("Rfrontalpole",)
    assert result.missing == "exclude"
    # The excluded region's density is dropped, not zeroed: the coupling dict
    # simply has no entry for it, so the engine adds no receptor drive there.
    assert "Rfrontalpole" not in result.coupling


def test_coupling_rejects_an_unknown_missing_policy(fixture_expression):
    with pytest.raises(ValueError, match="no 'zero' policy"):
        maps.whole_brain_coupling(
            fixture_expression, "DRD2", mechanism_gains=1.0, missing="zero"
        )


def test_coupling_rejects_an_empty_label_set(fixture_expression):
    with pytest.raises(ValueError, match="empty"):
        maps.whole_brain_coupling(
            fixture_expression, "DRD2", mechanism_gains=1.0, labels=[]
        )


def test_maps_module_is_reachable_from_the_human_package():
    assert human.whole_brain_coupling is maps.whole_brain_coupling
    assert human.HumanCoupling is maps.HumanCoupling
    assert human.MISSING_POLICIES == maps.MISSING_POLICIES
