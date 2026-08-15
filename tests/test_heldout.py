"""P8.2: the held-out compound records, and the guards on them.

The acceptance criterion this file exists for is "no held-out compound
silently moved into the training set". That is checked by reading the
training set off the evidence module rather than restating it, so a drug that
acquires a fitting target later starts failing here instead of quietly
passing.
"""

import numpy as np
import pytest

from neurocascade import fingerprint as fp
from neurocascade import heldout as ho
from neurocascade.calibration import behavioral_evidence as evidence


COMPOUNDS = sorted(ho.HELD_OUT)


# ---------------------------------------------------------------------------
# The acceptance criteria, as assertions
# ---------------------------------------------------------------------------


def test_at_least_six_held_out_compounds():
    """The brief's first acceptance criterion. Asserted rather than counted
    in prose, so a compound that fails to import cannot leave the claim
    standing."""
    assert len(ho.HELD_OUT) >= 6, sorted(ho.HELD_OUT)


def test_no_held_out_compound_is_in_the_training_set():
    """The criterion the whole exercise depends on. `training_compounds()`
    reads `behavioral_evidence` rather than restating a list, so this cannot
    be satisfied by forgetting to update a literal."""
    training = ho.training_compounds()
    assert training, "the training set came back empty, which is a bug here"
    overlap = training & set(ho.HELD_OUT)
    assert not overlap, "held-out compounds found in the training set: %s" % (
        sorted(overlap),
    )


def test_the_training_set_is_the_three_drugs_phase_1_built_pk_for():
    """Pins what the training set actually is, so the disjointness test above
    cannot pass because the evidence module quietly emptied."""
    assert ho.training_compounds() == {"diazepam", "fentanyl", "haloperidol"}


def test_the_compounds_span_more_than_one_mechanism_class():
    """"At least six compounds spanning different mechanisms" is the other
    half of the first criterion, and a set of six benzodiazepines would meet
    the count and not the intent."""
    classes = {c.mechanism_class for c in ho.HELD_OUT.values()}
    assert len(classes) >= 5, sorted(classes)


def test_no_held_out_compound_has_a_behavioural_evidence_entry():
    """A second, independent route to the same guarantee: not only is no
    held-out compound in the training set by name, none of them appears as
    the drug of any evidence entry at all, calibration-ready or not."""
    drugs = {entry.drug for entry in evidence.BEHAVIORAL_EVIDENCE}
    for name in ho.HELD_OUT:
        assert name not in drugs, name


# ---------------------------------------------------------------------------
# The record classes refuse what they are supposed to refuse
# ---------------------------------------------------------------------------


def test_a_ki_record_needs_exactly_one_of_a_value_and_a_ceiling():
    with pytest.raises(ValueError, match="exactly one"):
        ho.KiRecord("D2", source="x")
    with pytest.raises(ValueError, match="exactly one"):
        ho.KiRecord("D2", source="x", ki_nM=1.0, censored_above=10.0)


def test_an_uncited_affinity_is_refused():
    with pytest.raises(ValueError, match="uncited"):
        ho.KiRecord("D2", source="", ki_nM=1.0)


def test_a_neutral_antagonist_must_have_efficacy_exactly_zero():
    """Guards the one field whose sign decides the direction of every
    behavioural prediction this task makes."""
    with pytest.raises(ValueError, match="efficacy exactly 0"):
        ho.EfficacyRecord("D2", 0.3, "antagonist", "assay", "source")


def test_an_efficacy_above_the_scale_is_refused_with_the_likely_cause():
    with pytest.raises(ValueError, match="percentage"):
        ho.EfficacyRecord("D2", 84.0, "agonist", "assay", "source")


def test_an_efficacy_needs_a_functional_assay_and_a_citation():
    """A Ki table has no efficacy column, so an efficacy without a functional
    assay behind it is the number P8.1's MissingEfficacyError exists to
    refuse, arriving from the other side."""
    with pytest.raises(ValueError, match="functional assay"):
        ho.EfficacyRecord("D2", 1.0, "agonist", "", "source")


def test_occupancy_must_be_a_fraction_not_a_percentage():
    with pytest.raises(ValueError, match="fraction"):
        ho.OccupancyObservation("x", "D2", "rat", "IV", 1.0, occupancy=65.0,
                                method="m", source="s")


def test_a_tabulated_observation_without_a_ratio_is_a_contradiction():
    with pytest.raises(ValueError, match="contradiction"):
        ho.BehaviouralObservation(
            "x", "anxiety", "rat", "IP", 1.0, measure="m",
            direction="decrease", reporting="tabulated", source="s",
        )


def test_a_not_searched_record_may_not_claim_queries_and_vice_versa():
    """'We looked and found nothing' and 'nobody looked' are different facts.
    Conflating them would let an unrun search be reported as evidence about
    the literature, which is the most consequential overclaim available in
    this task."""
    with pytest.raises(ValueError, match="lists no queries"):
        ho.SearchRecord("x", queries=(), candidates=(), conclusion="c")
    with pytest.raises(ValueError, match="marked not-searched"):
        ho.SearchRecord("x", queries=("q",), candidates=(), conclusion="c",
                        searched=False)


def test_every_measured_ki_has_an_efficacy_and_no_efficacy_is_orphaned():
    with pytest.raises(ValueError, match="functional efficacy"):
        ho.HeldOutCompound(
            "x", "class", ki=[ho.KiRecord("D2", source="s", ki_nM=1.0)],
            efficacy=[], note="n",
        )
    with pytest.raises(ValueError, match="no measured Ki behind"):
        ho.HeldOutCompound(
            "x", "class", ki=[],
            efficacy=[ho.EfficacyRecord("D2", 0.0, "antagonist", "a", "s")],
            note="n",
        )


# ---------------------------------------------------------------------------
# Every record carries its provenance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", COMPOUNDS)
def test_every_affinity_cites_a_pmid_or_doi(name):
    """Rule 1 of this codebase, checked mechanically rather than trusted."""
    for record in ho.HELD_OUT[name].ki:
        assert "PMID" in record.source or "DOI" in record.source.upper() \
            or "PMC" in record.source, (name, record.target, record.source)


@pytest.mark.parametrize("name", COMPOUNDS)
def test_every_efficacy_cites_a_separate_functional_assay(name):
    for record in ho.HELD_OUT[name].efficacy.values():
        assert record.assay.strip(), (name, record.target)
        assert "PMID" in record.source or "PMC" in record.source, (
            name, record.target, record.source
        )


@pytest.mark.parametrize("name", COMPOUNDS)
def test_every_compound_documents_at_least_one_search(name):
    """A compound with no recorded gaps in a task this incomplete would mean
    nobody wrote the gaps down, not that there were none."""
    assert ho.HELD_OUT[name].searches, name


@pytest.mark.parametrize("name", COMPOUNDS)
def test_every_searched_record_lists_its_queries_and_a_conclusion(name):
    for record in ho.HELD_OUT[name].searches:
        assert record.conclusion.strip(), (name, record.what)
        if record.searched:
            assert record.queries, (name, record.what)


@pytest.mark.parametrize("name", COMPOUNDS)
def test_tested_inactive_and_never_tested_are_disjoint(name):
    """A target cannot be both screened-and-inactive and never-measured, and
    the distinction is what lets a compound's model be called complete over
    the targets it covers."""
    compound = ho.HELD_OUT[name]
    inactive = {t for t, _, _ in compound.tested_inactive}
    untested = {t for t, _ in compound.never_tested}
    assert not (inactive & untested), (name, sorted(inactive & untested))


@pytest.mark.parametrize("name", COMPOUNDS)
def test_no_target_is_both_measured_and_declared_inactive(name):
    compound = ho.HELD_OUT[name]
    measured = {r.target for r in compound.ki if not r.is_censored}
    inactive = {t for t, _, _ in compound.tested_inactive}
    assert not (measured & inactive), (name, sorted(measured & inactive))


# ---------------------------------------------------------------------------
# The fingerprint the records produce
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", COMPOUNDS)
def test_every_compound_builds_a_usable_fingerprint(name):
    fingerprint = ho.HELD_OUT[name].fingerprint()
    assert fingerprint.entries, name
    for target, entry in fingerprint.entries.items():
        assert target in fp.TARGETS
        assert entry.ki_nM > 0.0
        assert entry.reference, (name, target)
        assert entry.efficacy_source, (name, target)


@pytest.mark.parametrize("name", COMPOUNDS)
def test_a_censored_affinity_never_becomes_a_ligand(name):
    """This module's censored-Ki policy, enforced. Instantiating a ligand at
    an assay ceiling would assert an affinity the assay explicitly failed to
    detect, and at a target with endogenous tone it would manufacture a
    displacement that was measured not to happen."""
    compound = ho.HELD_OUT[name]
    fingerprint = compound.fingerprint()
    for record in compound.ki:
        if record.is_censored:
            assert record.target not in fingerprint.entries
            assert record.target in fingerprint.unmodelled


def test_alprazolam_is_the_only_compound_reaching_the_benzodiazepine_site():
    """Alprazolam shares its target with diazepam, which IS a training drug.
    That makes it the sharpest test in the set and also the place a training
    number could most easily leak in, so which compound touches that site is
    pinned."""
    reaching = [
        name for name, compound in ho.HELD_OUT.items()
        if "GABA_A_BZD" in compound.modelled_targets
    ]
    assert reaching == ["alprazolam"]


def test_alprazolams_affinity_is_not_diazepams():
    """Rule 2, at the one place in this task where substituting a training
    drug's measured value for a held-out drug's would let a prediction
    succeed. Diazepam's benzodiazepine-site Kd in this codebase is 3.6 nM
    ([Mohler1977]); alprazolam's must be its own number."""
    from neurocascade.calibration import endpoint_calibration as ec

    alprazolam = ho.HELD_OUT["alprazolam"].fingerprint()
    assert alprazolam.entries["GABA_A_BZD"].ki_nM != ec.DIAZEPAM_KI_NM


# ---------------------------------------------------------------------------
# The derived numbers, re-derived
# ---------------------------------------------------------------------------


def test_the_open_arm_conversion_reproduces_the_published_diazepam_ratio():
    """The conversion alprazolam's entries use, checked against the one
    published contrast this repository already carries a ratio for.
    [Graziani2022]: 30.7 s vehicle and 89.0 s diazepam of open-arm time in a
    300 s trial, which `endpoint_calibration` records as an anxiety ratio of
    0.7835. If this test fails, alprazolam's ratios are on a different scale
    from the repository's own and are not comparable with anything."""
    from neurocascade.calibration import endpoint_calibration as ec

    observation = ho.BehaviouralObservation.from_open_arm_percent(
        vehicle_percent=100.0 * 30.7 / 300.0, vehicle_sd=1.0,
        drug_percent=100.0 * 89.0 / 300.0, drug_sd=1.0,
        n_vehicle=10, n_drug=10,
        compound="diazepam-check", endpoint="anxiety", species="rat",
        route="IP", dose_mg_per_kg=2.0, measure="m", direction="decrease",
        reporting="tabulated", source="[Graziani2022] check",
    )
    published = ec.TARGETS_BY_KEY["anxiety/diazepam/Graziani2022"].levels[0][2]
    assert observation.ratio == pytest.approx(published, rel=1e-3)


def test_the_alprazolam_series_is_non_monotone_in_the_point_estimates():
    """The measured fact the monotonicity refutation rests on, asserted here
    on the DATA so that a change to the analysis cannot make it disappear."""
    series = sorted(
        (o for o in ho.HELD_OUT["alprazolam"].behavioural_observations
         if o.endpoint == "anxiety" and o.is_quantitative),
        key=lambda o: o.dose_mg_per_kg,
    )
    assert [o.dose_mg_per_kg for o in series] == [0.1, 0.3, 1.0]
    ratios = [o.ratio for o in series]
    assert ratios[1] < ratios[0]
    assert ratios[2] > ratios[1]


def test_the_open_arm_conversion_uses_the_standard_error_not_the_deviation():
    """A ratio of group MEANS needs the uncertainty of the means. Using the
    group standard deviation instead would inflate every dispersion by
    sqrt(n) and make every residual look smaller than it is."""
    kwargs = dict(
        compound="x", endpoint="anxiety", species="rat", route="IP",
        dose_mg_per_kg=1.0, measure="m", direction="decrease",
        reporting="tabulated", source="s",
    )
    small = ho.BehaviouralObservation.from_open_arm_percent(
        20.0, 10.0, 30.0, 10.0, n_vehicle=10, n_drug=10, **kwargs)
    large = ho.BehaviouralObservation.from_open_arm_percent(
        20.0, 10.0, 30.0, 10.0, n_vehicle=40, n_drug=40, **kwargs)
    assert small.ratio == pytest.approx(large.ratio)
    assert large.ratio_sd == pytest.approx(small.ratio_sd / 2.0, rel=1e-6)


def test_an_open_arm_time_of_a_hundred_percent_is_refused():
    with pytest.raises(ValueError, match="no avoidance"):
        ho.BehaviouralObservation.from_open_arm_percent(
            100.0, 1.0, 50.0, 1.0, n_vehicle=5, n_drug=5,
            compound="x", endpoint="anxiety", species="rat", route="IP",
            dose_mg_per_kg=1.0, measure="m", direction="decrease",
            reporting="tabulated", source="s",
        )
