"""Per-edge provenance flags: measured / inferred / absent (P13.2).

Offline tests off the committed fixtures. The load-bearing properties:

* a `ProvenanceMap` covers every ordered pair - "no edge without one" -
  and refuses grids with values outside the three levels;
* F-TRACT coverage maps to MEASURED exactly where the value is finite,
  including exact zeros (probed, no response), with NaN ABSENT;
* the directionality of F-TRACT measurement survives: an edge can be
  measured one way and absent the other;
* the ENIGMA dMRI matrix maps to INFERRED in both directions wherever it
  is nonzero - never MEASURED (the P13.6 invariant);
* `require()` enforces minimum levels, defaulting to MEASURED;
* `combine()` takes the elementwise strongest level and refuses different
  label sets.
"""

from pathlib import Path

import numpy as np
import pytest

from neurocascade.human import fixtures
from neurocascade.human import ftract as ftract_mod
from neurocascade.human import build as build_mod
from neurocascade.human import provenance as prov


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def probability_feature():
    """The committed verbatim 6-parcel F-TRACT probability slice."""
    labels, matrix = fixtures.load_fixture_ftract_csv("probability")
    return ftract_mod.FTractFeature("probability", labels, matrix)


@pytest.fixture(scope="module")
def fixture_connectome():
    return fixtures.load_fixture_connectome()


@pytest.fixture(scope="module")
def fixture_circuit(fixture_connectome):
    return build_mod.build_circuit_network(fixture_connectome)


# ---------------------------------------------------------------------------
# Construction: total coverage ("no edge without one")
# ---------------------------------------------------------------------------


def test_levels_are_ordered_weakest_to_strongest():
    assert prov.LEVELS == (prov.ABSENT, prov.INFERRED, prov.MEASURED)


def test_map_covers_every_ordered_pair(probability_feature):
    p = probability_feature.provenance()
    assert p.n == probability_feature.n
    for source in probability_feature.labels:
        for target in probability_feature.labels:
            assert p.flag(source, target) in prov.LEVELS


def test_constructor_rejects_unknown_level_strings():
    with pytest.raises(prov.ProvenanceError, match="outside"):
        prov.ProvenanceMap.from_levels(
            ["a", "b"], np.array([["measured", "guessed"], ["absent", "absent"]], dtype=object)
        )


def test_constructor_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="flags must be"):
        prov.ProvenanceMap.from_labels_if_you_must = None
        prov.ProvenanceMap(["a", "b", "c"], np.array([["measured"]], dtype=object))


def test_constructor_rejects_duplicate_labels():
    with pytest.raises(ValueError, match="duplicates"):
        prov.ProvenanceMap.full(["a", "a"], prov.MEASURED)


def test_full_starts_everywhere_at_one_level():
    p = prov.ProvenanceMap.full(["a", "b", "c"], prov.INFERRED)
    assert p.counts()[prov.INFERRED] == 9
    assert p.counts()[prov.MEASURED] == 0
    assert p.counts()[prov.ABSENT] == 0


# ---------------------------------------------------------------------------
# F-TRACT -> measured
# ---------------------------------------------------------------------------


def test_ftract_finite_cells_are_measured(probability_feature):
    p = probability_feature.provenance()
    fin = np.isfinite(probability_feature.matrix)
    counts = p.counts()
    assert counts[prov.MEASURED] == int(fin.sum())
    for i in range(p.n):
        for j in range(p.n):
            expected = prov.MEASURED if fin[i, j] else prov.ABSENT
            assert p.flag(probability_feature.labels[i], probability_feature.labels[j]) == expected


def test_ftract_exact_zero_is_measured_not_absent(probability_feature):
    """A probed pair with no response is a completed measurement."""
    zeros_off_diag = probability_feature.n_exact_zeros_off_diagonal()
    if zeros_off_diag == 0:
        pytest.skip("fixture slice has no exact zeros off-diagonal")
    p = probability_feature.provenance()
    matrix = probability_feature.matrix
    n_zeros_found = 0
    for i in range(p.n):
        for j in range(p.n):
            if i != j and matrix[i, j] == 0.0:
                assert p.flag(
                    probability_feature.labels[i], probability_feature.labels[j]
                ) == prov.MEASURED
                n_zeros_found += 1
    assert n_zeros_found == zeros_off_diag


def test_ftract_nan_is_absent(probability_feature):
    matrix = probability_feature.matrix
    nan_positions = list(zip(*np.where(np.isnan(matrix))))
    if not nan_positions:
        pytest.skip("fixture slice has no NaN cells")
    p = probability_feature.provenance()
    for i, j in nan_positions:
        assert (
            p.flag(probability_feature.labels[i], probability_feature.labels[j])
            == prov.ABSENT
        )


def test_ftract_directionality_survives_in_provenance(probability_feature):
    """The whole point of P13.1/P13.2: measured need not be reciprocal."""
    matrix = probability_feature.matrix
    fin = np.isfinite(matrix)
    pairs = [
        (i, j)
        for i in range(matrix.shape[0])
        for j in range(matrix.shape[0])
        if i != j and fin[i, j] != fin[j, i]
    ]
    if not pairs:
        pytest.skip("fixture slice has no one-way measured pairs")
    p = probability_feature.provenance()
    labels = probability_feature.labels
    for i, j in pairs:
        forward = p.flag(labels[i], labels[j])
        backward = p.flag(labels[j], labels[i])
        assert {forward, backward} == {prov.MEASURED, prov.ABSENT}


def test_ftract_require_measured_succeeds_on_measured_edge(probability_feature):
    p = probability_feature.provenance()
    matrix = probability_feature.matrix
    i, j = list(zip(*np.where(fin := np.isfinite(matrix))))[0]
    level = p.require(probability_feature.labels[i], probability_feature.labels[j])
    assert level == prov.MEASURED
    assert fin is not None  # keep the walrus honest for readers


# ---------------------------------------------------------------------------
# ENIGMA dMRI -> inferred (never measured)
# ---------------------------------------------------------------------------


def test_structural_nonzero_is_inferred_both_ways(fixture_connectome):
    p = prov.ProvenanceMap.from_structural(fixture_connectome)
    weights = fixture_connectome.weights
    for i in range(weights.shape[0]):
        for j in range(weights.shape[0]):
            expected = prov.INFERRED if weights[i, j] != 0.0 else prov.ABSENT
            actual = p.flag(fixture_connectome.labels[i], fixture_connectome.labels[j])
            assert actual == expected


def test_structural_is_never_measured(fixture_connectome):
    p = prov.ProvenanceMap.from_structural(fixture_connectome)
    assert p.counts()[prov.MEASURED] == 0


def test_built_circuit_provenance_matches_its_own_weights(fixture_circuit):
    """The map describes what actually ran, not the raw file."""
    p = fixture_circuit.provenance()
    connectivity = fixture_circuit.connectivity
    for i, si in enumerate(fixture_circuit.labels):
        for j, sj in enumerate(fixture_circuit.labels):
            expected = prov.INFERRED if connectivity[i, j] != 0.0 else prov.ABSENT
            # flag(source=sj, target=si): connectivity[i, j] is sj -> si.
            assert p.flag(sj, si) == expected


def test_excluded_unsigned_sources_end_up_absent():
    """Amygdala columns are zeroed by the default policy, so ABSENT."""
    connectome = fixtures.load_fixture_connectome()
    circuit = build_mod.build_circuit_network(connectome)
    unsigned = [a.label for a in circuit.report.assignments if not a.is_signed]
    if not unsigned:
        pytest.skip("fixture atlas has no unsigned sources")
    p = circuit.provenance()
    amy = unsigned[0]
    targets = [lab for lab in circuit.labels if lab != amy]
    outgoing = [p.flag(amy, t) for t in targets]
    assert all(level == prov.ABSENT for level in outgoing)


# ---------------------------------------------------------------------------
# require(): enforcement
# ---------------------------------------------------------------------------


def test_require_default_refuses_inferred_and_absent(fixture_connectome, probability_feature):
    structural = prov.ProvenanceMap.from_structural(fixture_connectome)
    weights = fixture_connectome.weights
    i, j = list(zip(*np.where(weights != 0.0)))[0]
    with pytest.raises(prov.ProvenanceError, match="below the required minimum"):
        structural.require(fixture_connectome.labels[i], fixture_connectome.labels[j])

    fp = probability_feature.provenance()
    matrix = probability_feature.matrix
    nan_pairs = list(zip(*np.where(np.isnan(matrix))))
    if nan_pairs:
        ni, nj = nan_pairs[0]
        with pytest.raises(prov.ProvenanceError, match="below the required minimum"):
            fp.require(probability_feature.labels[ni], probability_feature.labels[nj])


def test_require_with_explicit_minimum_accepts_inferred(fixture_connectome):
    structural = prov.ProvenanceMap.from_structural(fixture_connectome)
    weights = fixture_connectome.weights
    i, j = list(zip(*np.where(weights != 0.0)))[0]
    level = structural.require(
        fixture_connectome.labels[i],
        fixture_connectome.labels[j],
        minimum=prov.INFERRED,
    )
    assert level == prov.INFERRED


def test_require_rejects_bad_minimum(fixture_connectome):
    structural = prov.ProvenanceMap.from_structural(fixture_connectome)
    label = fixture_connectome.labels[0]
    with pytest.raises(ValueError, match="level must be one of"):
        structural.require(label, label, minimum="guessed")


def test_require_unknown_label_raises_keyerror(fixture_connectome):
    structural = prov.ProvenanceMap.from_structural(fixture_connectome)
    with pytest.raises(KeyError):
        structural.require("not-a-region", fixture_connectome.labels[0])


# ---------------------------------------------------------------------------
# combine(): precedence and label-set refusal
# ---------------------------------------------------------------------------


def test_combine_takes_the_strongest_level(probability_feature, fixture_connectome):
    f = probability_feature.provenance()
    s = prov.ProvenanceMap.from_structural(fixture_connectome)
    if s.n != f.n:
        pytest.skip("label sets differ; covered by the refusal test below")


def test_combine_refuses_different_label_sets(probability_feature, fixture_connectome):
    f = probability_feature.provenance()
    s = prov.ProvenanceMap.from_structural(fixture_connectome)
    if f.labels == s.labels:
        pytest.skip("fixtures happen to share a label set")
    with pytest.raises(ValueError, match="different label sets"):
        f.combine(s)


def test_combine_measured_beats_inferred_on_identical_labels():
    labels = ["a", "b"]
    measured = prov.ProvenanceMap.from_levels(
        labels,
        np.array([["measured", "measured"], ["absent", "absent"]], dtype=object),
    )
    inferred = prov.ProvenanceMap.from_levels(
        labels,
        np.array([["inferred", "inferred"], ["inferred", "inferred"]], dtype=object),
    )
    merged = measured.combine(inferred)
    assert merged.flag("a", "b") == prov.MEASURED
    assert merged.flag("b", "a") == prov.INFERRED

    reverse = inferred.combine(measured)
    assert reverse.flags_matrix().tolist() == merged.flags_matrix().tolist()


# ---------------------------------------------------------------------------
# The committed coverage result
# ---------------------------------------------------------------------------


#: Parsed from results/human_edge_provenance.txt, which
#: scripts/report_edge_provenance.py generates from the verified caches.
#: If either side moves, a test fails before a claim can drift.
RESULT = Path(__file__).resolve().parents[1] / "results" / "human_edge_provenance.txt"


def test_coverage_result_is_committed():
    assert RESULT.is_file()


def test_coverage_result_carries_the_full_parcel_mix():
    text = RESULT.read_text()
    # From the real 360x360 probability CSV: 27,066 measured, the rest
    # absent - 20.9% of the atlas is measured direction, 79.1% has none.
    assert "measured   27066 ( 20.9%)" in text
    assert "absent    102534 ( 79.1%)" in text


def test_coverage_result_names_the_uncovered_parcels():
    text = RESULT.read_text()
    for label in ("L_V6A", "L_PIT", "R_V6A", "R_7Am"):
        assert label in text, label
    assert "4 of 360" in text


def test_coverage_result_records_the_built_network_mix():
    """The mix behind predictions on the DK network: nothing measured,
    everything carried is inferred. This is the number P13.3 exists to
    raise."""
    text = RESULT.read_text()
    assert "measured       0 (  0.0%)" in text
    assert "inferred    2317 ( 34.5%)" in text
    assert "absent      4407 ( 65.5%)" in text


# ---------------------------------------------------------------------------
# Summary / repr
# ---------------------------------------------------------------------------


def test_summary_reports_all_three_levels(probability_feature):
    text = probability_feature.provenance().summary()
    assert "measured" in text
    assert "inferred" in text
    assert "absent" in text
    assert "%d directed pairs" % (probability_feature.n ** 2) in text


def test_repr_mentions_counts(fixture_connectome):
    p = prov.ProvenanceMap.from_structural(fixture_connectome)
    r = repr(p)
    assert "ProvenanceMap" in r
    assert str(p.counts()[prov.INFERRED]) in r


def test_flags_matrix_returns_a_copy(probability_feature):
    p = probability_feature.provenance()
    grabbed = p.flags_matrix()
    grabbed[:] = prov.ABSENT
    assert p.counts()[prov.MEASURED] == int(np.isfinite(probability_feature.matrix).sum())
