"""P4.3: the receptor maps against independently known distributions.

Everything up to the `real_data` block runs offline off the committed
fixtures, including every one of the eighteen cited claims over six
receptors. That is possible because P4.3 added the seven regions its
sources name to the shared fixture ontology
(`connectome.ANATOMY_ONLY_REGION_IDS`) and recut the expression fixtures
over them, and because a claim's outcome depends only on the rows of the
regions it names - the fixture rows are byte-identical slices of the
verified Allen responses, so "from the fixture" is the same number and not
an approximation of it. `test_real_atlas_gives_the_same_verdicts` is what
says so rather than leaving it as a claim in a docstring.

The framing these tests inherit from `anatomy.py`: a failing concordant
claim is an ingestion or normalization bug, not a finding about the brain.
"""

import numpy as np
import pytest

from neurocascade import scenarios
from neurocascade.legacy import connectome, expression
from neurocascade.legacy.expression import (
    anatomy,
    fetch,
    fixtures,
    mismatch,
    sources,
)


@pytest.fixture
def ontology():
    return connectome.load_fixture_ontology()


@pytest.fixture
def atlas(ontology):
    return fixtures.load_fixture_atlas(
        symbols=list(anatomy.VALIDATED_SYMBOLS), ontology=ontology
    )


@pytest.fixture
def results(atlas, ontology):
    return anatomy.check_atlas(atlas, ontology=ontology)


# ---------------------------------------------------------------------------
# The acceptance criterion: five receptors, cited, checked
# ---------------------------------------------------------------------------


def test_at_least_five_receptors_are_validated():
    assert len(anatomy.VALIDATED_SYMBOLS) >= 5
    assert set(anatomy.VALIDATED_SYMBOLS) == {
        "Drd1", "Drd2", "Oprm1", "Gabra1", "Cnr1", "Adora2a",
    }
    # The roadmap's P4.3 line names five distributions; every gene in it has
    # a claim here, and each of the three the model already simulates
    # (Oprm1, Drd2, Gabra1) is among them.
    for symbol in ("Drd1", "Drd2", "Oprm1", "Gabra1", "Cnr1", "Adora2a"):
        assert anatomy.CLAIMS_BY_SYMBOL[symbol]


def test_every_claim_carries_a_citation_with_species_and_modality():
    for claim in anatomy.CLAIMS:
        source = claim.source
        assert source.key in anatomy.ANATOMY_SOURCES
        assert source.citation and source.species and source.method
        assert source.modality in ("mRNA", "protein")
        # The expectation itself has to be quoted or paraphrased from the
        # source, not asserted: an uncited expectation is an invented number
        # wearing an anatomy costume.
        assert len(claim.evidence) > 40


def test_every_source_is_mouse_or_says_in_its_record_why_not():
    """Rule 2 applied to anatomical expectations rather than to fits.

    Four of the six receptors are validated against mouse sources only.
    Cnr1 is the exception and both of its non-mouse sources say so in their
    `note`, name what they are used for, and supply no number to the map.
    """
    non_mouse = [
        source
        for source in anatomy.ANATOMY_SOURCES.values()
        if "Mus musculus" not in source.species
    ]
    assert [s.key for s in non_mouse] == ["mailleux1992", "herkenham1991"]
    for source in non_mouse:
        assert "RAT" in source.note
    mouse_only_symbols = {
        symbol
        for symbol, claims in anatomy.CLAIMS_BY_SYMBOL.items()
        if all("Mus musculus" in c.source.species for c in claims)
    }
    assert mouse_only_symbols == {"Drd1", "Drd2", "Oprm1", "Gabra1", "Adora2a"}


def test_every_asserted_claim_behaves_as_its_record_says(results):
    """The validation itself. A failure here is a bug in the ingestion or
    the normalization, and the right response is to find the defect."""
    asserted = anatomy.asserted(results)
    assert len(asserted) >= 12
    for result in asserted:
        assert result.ok is True, (
            "%s: %s" % (result.claim.describe(), result.summary())
        )


def test_the_reported_claims_are_reported_and_not_hidden(results):
    """Single-tier claims are not asserted (see anatomy.py rule 3), so this
    is what stops that rule from becoming a place to put inconvenient
    results: their outcomes are pinned here explicitly, three holding and
    one not, and any change to any of the four fails."""
    reported = {
        r.claim.describe(): r.holds for r in results if not r.claim.assertable
    }
    assert len(reported) == 4
    holds = [describe for describe, ok in reported.items() if ok]
    fails = [describe for describe, ok in reported.items() if not ok]
    assert len(holds) == 3 and len(fails) == 1
    assert fails[0].startswith("Oprm1: MOp above CBX")


def test_the_one_disagreement_is_two_regions_at_the_floor(results, atlas, ontology):
    """The root cause of the single unexpected outcome, measured.

    Erbs et al. score mouse cortical MOR at + and cerebellar cortex at -,
    one tier apart at the bottom of their scale. The transcript map puts
    the two within 1% of each other, both in the bottom eighth of the
    whole-brain Oprm1 distribution: there is no ordering in the map to be
    right or wrong about. It is not a mis-join - a mis-join would move a
    region by orders of magnitude, as this same gene's MH does in the other
    direction - and it disappears entirely under plane='any'.
    """
    failed = anatomy.failures(results)
    assert len(failed) == 1
    result = failed[0]
    assert 0.98 < result.ratio < 1.02

    energy = atlas.energy_matrix(
        [int(ontology.id_for_acronym(a)) for a in ("MOp", "CBX", "MH")],
        symbols=["Oprm1"],
        plane=anatomy.DEFAULT_PLANE,
        combine=anatomy.DEFAULT_COMBINE,
    )
    mop, cbx, mh = energy.for_gene("Oprm1")
    assert abs(mop - cbx) / max(mop, cbx) < 0.02
    assert mh > 40 * max(mop, cbx)


def test_the_disagreement_is_inside_the_between_experiment_spread(atlas, ontology):
    """Pooling Oprm1's three experiments flips it, which is the size of the
    thing: P4.1 measured Oprm1's median between-experiment ratio at 6.2, and
    a 1% difference is far inside that."""
    pooled = anatomy.check_atlas(
        atlas, plane="any", combine="mean", ontology=ontology
    )
    assert anatomy.failures(pooled) == []


def test_expected_failures_are_asserted_to_fail(results):
    """A divergence that stops diverging is as much a signal as one that
    starts. Both of these are genes `mismatch.py` already flags."""
    divergent = [r for r in results if not r.claim.expected_to_hold]
    assert [r.claim.symbol for r in divergent] == ["Drd1", "Cnr1"]
    for result in divergent:
        assert result.holds is False
        assert result.ok is True
        assert result.claim.mechanism and result.claim.mechanism_source
    # And the terminal-field mechanism is the one mismatch.py records.
    assert anatomy.mismatch_context("Cnr1").severity == "terminal-field"
    assert "Cnr1" in mismatch.TERMINAL_FIELD_SYMBOLS


def test_the_cnr1_divergence_has_a_measured_size(results):
    """P4.2's discipline: a divergence gets a number, not an adjective."""
    cnr1 = [
        r for r in results
        if r.claim.symbol == "Cnr1" and not r.claim.expected_to_hold
    ][0]
    # Herkenham et al. put SNr and GPe binding ABOVE the caudoputamen; the
    # transcript puts them at a quarter to a third of it.
    assert 0.15 < cnr1.ratio < 0.30
    assert cnr1.values["SNr"] < cnr1.values["CP"]
    assert cnr1.values["GPe"] < cnr1.values["CP"]


def test_a_claim_expected_to_fail_must_carry_its_mechanism():
    with pytest.raises(ValueError) as caught:
        anatomy.Claim(
            symbol="Oprm1", high=("CP",), low=("MOp",),
            source=anatomy.ERBS2015, separation=2.0,
            evidence="x" * 50, expected_to_hold=False,
        )
    assert "dropped inconvenient result" in str(caught.value)


def test_a_source_must_say_what_it_measured():
    with pytest.raises(ValueError) as caught:
        anatomy.AnatomySource(
            key="x", citation="y", species="Mus musculus",
            method="z", modality="binding",
        )
    assert "mRNA" in str(caught.value)


# ---------------------------------------------------------------------------
# The checks that catch the failure modes an anatomy check exists to catch
# ---------------------------------------------------------------------------


def test_the_striatal_markers_are_striatal_and_gabra1_is_not(results):
    """The sharpest single test in the module, stated as a test.

    Drd1, Drd2 and Adora2a peak in the striatum and Gabra1's alpha1 is
    scored FAINT there by the one mouse ISH source, with the pallidum and
    substantia nigra strong instead. A gene row shifted by one column would
    have to survive both halves of that to go unnoticed.
    """
    by_symbol = {}
    for result in results:
        by_symbol.setdefault(result.claim.symbol, []).append(result)

    for symbol in ("Drd1", "Drd2", "Adora2a"):
        striatal = by_symbol[symbol][0]
        assert set(striatal.claim.high) <= {"CP", "ACB", "OT"}
        assert striatal.holds is True

    cortical, pallidal = by_symbol["Gabra1"][0], by_symbol["Gabra1"][1]
    assert cortical.values["CP"] < cortical.values["MOp"]
    assert pallidal.values["CP"] < pallidal.values["GPe"]


def test_drd2_is_midbrain_weighted_relative_to_drd1(atlas, ontology):
    """The one cross-gene check, in the only shape a cross-gene check can
    legitimately take: a ratio of within-gene ratios, so the per-experiment
    probe difference that makes energies incomparable between genes cancels
    rather than being ignored."""
    energy = atlas.energy_matrix(
        [int(ontology.id_for_acronym(a)) for a in ("VTA", "SNc", "CP")],
        symbols=["Drd1", "Drd2"],
        plane=anatomy.DEFAULT_PLANE,
        combine=anatomy.DEFAULT_COMBINE,
    )
    contrast = anatomy.midbrain_dopamine_contrast(energy)
    assert contrast["VTA"] > 10.0
    assert contrast["SNc"] > 10.0


def test_an_unmeasured_region_is_not_evidence_either_way(ontology):
    """NaN is not zero, one layer up. Htr2a is the fixture's sagittal-only
    gene, so under plane='coronal' its whole row is NaN; a claim over it
    must come back `None`, not `False`."""
    atlas = fixtures.load_fixture_atlas(symbols=["Htr2a"], ontology=ontology)
    claim = anatomy.Claim(
        symbol="Htr2a", high=("CP",), low=("MOp",),
        source=anatomy.HORTNAGL2013, separation=2.0,
        evidence="a deliberately contrived claim, used only to exercise the "
                 "unmeasured path on a real all-NaN row",
    )
    results = anatomy.check_atlas(
        atlas, plane="coronal", combine="mean", ontology=ontology,
        claims=[claim],
    )
    assert results[0].holds is None
    assert results[0].ok is None
    assert set(results[0].missing) == {"CP", "MOp"}


def test_claims_cannot_put_a_region_on_both_sides():
    with pytest.raises(ValueError):
        anatomy.Claim(
            symbol="Drd1", high=("CP",), low=("CP", "MOp"),
            source=anatomy.YOO2010_SOURCE, separation=2.0,
            evidence="x" * 50,
        )


def test_every_anatomy_only_fixture_region_is_named_by_a_claim(ontology):
    """The ontology grew for these tests, so the tests have to justify every
    region it grew by. SNc and CBX were already there for P3.2's sign
    table, so they are exempt from the converse check."""
    named = set(anatomy.claim_regions())
    for structure_id in connectome.ANATOMY_ONLY_REGION_IDS:
        acronym = ontology.acronym_for_id(structure_id)
        assert acronym in named, acronym


def test_the_policy_does_not_change_the_asserted_conclusions(atlas, ontology):
    """P4.1 deferred the plane and combine choice and P4.2 priced it for a
    correlation. This prices it for these claims: no policy changes any
    ASSERTED claim's outcome, and the only policy-dependent outcome in the
    sweep is the single-tier one already pinned above."""
    agreement = anatomy.policy_agreement(atlas, ontology=ontology)
    assert len(agreement) == 9
    assert set(counts["unexpected"] for counts in agreement.values()) == {0, 1}
    for (plane, combine), counts in agreement.items():
        assert counts["unmeasured"] == 0
        results = anatomy.check_atlas(
            atlas, plane=plane, combine=combine, ontology=ontology
        )
        for result in anatomy.asserted(results):
            assert result.ok is True, (plane, combine, result.summary())


# ---------------------------------------------------------------------------
# The real files. Extras: they skip when the gitignored cache is absent.
# ---------------------------------------------------------------------------

_REAL_CACHE_PRESENT = fetch.is_cached(sources.DATASET_INDEX_KEY) and all(
    fetch.is_cached(sources.unionize_key(symbol))
    for symbol in anatomy.VALIDATED_SYMBOLS
)

real_data = pytest.mark.skipif(
    not _REAL_CACHE_PRESENT,
    reason="data/expression/ is not populated; run scripts/fetch_expression.py",
)


@real_data
def test_real_atlas_gives_the_same_verdicts_as_the_fixture(results):
    """What makes the offline suite a real validation rather than a
    validation of the fixture: a claim's value depends only on the rows of
    the regions it names, and those rows are byte-identical between the
    fixture and the 7.5 MB cache."""
    ontology = connectome.load_ontology()
    atlas = expression.load_expression_atlas(
        ontology=ontology, symbols=list(anatomy.VALIDATED_SYMBOLS)
    )
    real = anatomy.check_atlas(atlas, ontology=ontology)
    assert len(real) == len(results)
    for from_cache, from_fixture in zip(real, results):
        assert from_cache.claim is from_fixture.claim
        assert from_cache.holds == from_fixture.holds
        assert from_cache.values == from_fixture.values


@real_data
def test_real_whole_brain_peaks_are_the_regions_the_sources_name():
    """The claims are pairwise and this is the whole-brain view of the same
    thing: over all 291 connectome regions, the top of each map. A pairwise
    check cannot see a region that is wrongly the highest in the brain."""
    ontology = connectome.load_ontology()
    matrix = connectome.load_connectome(ontology=ontology)
    atlas = expression.load_expression_atlas(
        ontology=ontology, symbols=list(anatomy.VALIDATED_SYMBOLS)
    )
    energy = atlas.for_connectome(
        matrix, plane=anatomy.DEFAULT_PLANE, combine=anatomy.DEFAULT_COMBINE
    )
    acronyms = energy.acronyms()

    def top(symbol, n):
        row = energy.for_gene(symbol)
        order = np.argsort(np.where(np.isfinite(row), row, -np.inf))[::-1]
        return [acronyms[i] for i in order[:n]]

    assert top("Oprm1", 1) == ["MH"]
    assert top("Adora2a", 1) == ["CP"]
    for symbol in ("Drd1", "Drd2", "Adora2a"):
        assert "CP" in top(symbol, 3)
    # Gabra1's peak is isocortical, and the striatum is nowhere near it.
    assert set(top("Gabra1", 5)) <= {
        "SSp-un", "SSp-m", "MOs", "SSp-bfd", "SSp-n", "SSp-ul", "MOp",
    }
    assert "CP" not in top("Gabra1", 50)


@real_data
def test_real_claustrum_is_not_a_striatal_smear():
    """The one conspicuous oddity of these maps, run down rather than left.

    CLA is the single highest region of the Drd1 and Drd2 transcript maps,
    above CP, and it is second for Adora2a. The cheap explanation would be
    partial-volume contamination: the claustrum is a thin sheet next to the
    striatum and the ISH grid is 200 um, so CLA voxels could be carrying
    striatal signal. That hypothesis makes a testable prediction - ONE
    mixing fraction should explain CLA across genes, because the geometry
    is one geometry - and the data refutes it: the implied fraction ranges
    from about 0 (Chrna4, Htr2a, Adrb1: CLA looks cortical) to about 1
    (Adra1d, Drd2: CLA looks striatal). Independently, Yoo et al. 2010
    measured mouse claustral D1 at 87.3 fmol/mg, fourth of their fifteen
    regions and 3.8x motor cortex, so a high claustral D1 is corroborated
    in protein. Recorded as an open question about claustral biology, not
    as an ingestion bug, and the Drd1 CLA>MOp claim is asserted because of
    the Yoo measurement.
    """
    ontology = connectome.load_ontology()
    atlas = expression.load_expression_atlas(ontology=ontology)
    ids = [int(ontology.id_for_acronym(a)) for a in ("CLA", "CP", "AIv")]
    energy = atlas.energy_matrix(
        ids, plane=anatomy.DEFAULT_PLANE, combine=anatomy.DEFAULT_COMBINE
    )
    fractions = []
    for symbol in energy.symbols:
        cla, cp, ctx = energy.for_gene(symbol)
        if not np.all(np.isfinite([cla, cp, ctx])):
            continue
        if abs(cp - ctx) < 0.3 * max(cp, ctx):
            continue  # the gene does not distinguish the two endpoints
        fractions.append((cla - ctx) / (cp - ctx))
    fractions = np.array(fractions)
    assert fractions.size >= 25
    assert fractions.min() < 0.2
    assert fractions.max() > 0.9
    # A single-geometry mixture would give a tight distribution; this one
    # spans the whole range, which is what refutes it.
    assert np.percentile(fractions, 75) - np.percentile(fractions, 25) > 0.4


@real_data
def test_real_report_matches_the_claims():
    """The committed report is generated from these records, so a claim
    edited without regenerating it would leave the two disagreeing."""
    import os

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
        "receptor_map_validation.txt",
    )
    text = open(path).read()
    assert "%d claims over %d receptors" % (
        len(anatomy.CLAIMS), len(anatomy.VALIDATED_SYMBOLS)
    ) in text
    for source in anatomy.ANATOMY_SOURCES.values():
        assert source.citation in text
    for node, structure_id in scenarios.MU_OPIOID_MULTI_REGION_STRUCTURES.items():
        assert node in text
