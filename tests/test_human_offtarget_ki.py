"""P23.1: human off-target binding fingerprints (haloperidol, risperidone).

Checks the three acceptance criteria:
  * per-drug human Ki at each named off-target, with assay species and source;
  * non-human values refused by the EXISTING species gate, not silently kept;
  * cells that could not be sourced recorded as searched-and-empty.
"""
import pytest

from neurocascade import species_gate
from neurocascade.human import offtarget_ki as ot


def test_the_five_targets_and_two_drugs_are_the_phase_23_set():
    assert ot.TARGETS == ("5HT2A", "M1", "H1", "alpha1", "D3")
    assert ot.DRUGS == ("haloperidol", "risperidone")


def test_every_grid_cell_has_a_human_value():
    """Two drugs x five targets = ten cells; all sourced, none empty."""
    assert ot.missing_cells() == set()
    assert ot.cells_with_human_value() == {
        (d, t) for d in ot.DRUGS for t in ot.TARGETS
    }


def test_every_human_record_carries_species_preparation_and_source():
    for rec in ot.HUMAN_KI:
        assert rec.is_human
        assert species_gate.normalize_species(rec.species) == species_gate.HUMAN
        assert rec.preparation            # the assay preparation is recorded
        assert rec.reference              # the primary reference is recorded
        assert rec.value_as_printed
        assert rec.selection              # why this row was registered
        assert rec.value_nM > 0


def test_the_three_loadbearing_primaries_carry_resolved_pmids():
    pmids = {r.reference: r.pmid for r in ot.HUMAN_KI if r.pmid}
    assert pmids["Richelson E, Souder T. 2000"] == "11132243"
    assert pmids["Schotte A, et al.,  1996"] == "8935801"
    assert pmids["Kroeze WK, et.al., 2003"] == "12629531"


def test_human_tissue_values_come_from_richelson_souder_2000():
    """Richelson & Souder 2000 is human post-mortem brain: the gold-standard
    provenance, present for 5-HT2A/H1/alpha-1 and (per SEARCHED_AND_EMPTY) absent
    for D3/M1."""
    tissue = [r for r in ot.HUMAN_KI if r.provenance_class == "human tissue"]
    assert tissue, "expected human-tissue rows"
    assert all(r.reference == "Richelson E, Souder T. 2000" for r in tissue)
    cells = {(r.drug, r.target) for r in tissue}
    assert cells == {
        (d, t) for d in ot.DRUGS for t in ("5HT2A", "H1", "alpha1")
    }
    # D3 and M1 have no tissue row for either drug.
    assert not any(r.target in ("D3", "M1") for r in tissue)


def test_the_gate_refuses_every_non_human_row():
    """The existing species_gate, not a re-implementation, must refuse each one,
    and the refusal must name the offending species."""
    assert ot.NON_HUMAN_EXCLUDED
    seen_species = set()
    for rec in ot.NON_HUMAN_EXCLUDED:
        assert not rec.is_human
        seen_species.add(species_gate.normalize_species(rec.species))
        with pytest.raises(species_gate.UnsourcedParameterError) as exc:
            ot.gate_record(rec)
        assert species_gate.normalize_species(rec.species) in str(exc.value)
    # rat, guinea pig, pig, bovine and mouse are all demonstrated as refused.
    assert {"rat", "guinea pig", "pig", "bovine", "mouse"} <= seen_species


def test_the_gate_passes_every_human_row():
    for rec in ot.HUMAN_KI:
        assert ot.gate_record(rec) is None


def test_schotte_1996_mixed_species_table_is_split_by_the_gate():
    """Schotte 1996 is the canonical risperidone paper and the clearest mixed
    table: human-cloned for 5-HT2A and D3, rodent/guinea-pig for alpha-1 and H1.
    The human rows are kept, the non-human ones refused -- from one paper."""
    schotte = [
        r for r in ot.HUMAN_KI + ot.NON_HUMAN_EXCLUDED
        if r.reference == "Schotte A, et al.,  1996"
    ]
    human = {(r.drug, r.target) for r in schotte if r.is_human}
    nonhuman = {(r.drug, r.target) for r in schotte if not r.is_human}
    assert ("risperidone", "5HT2A") in human
    assert ("risperidone", "D3") in human
    assert ("risperidone", "alpha1") in nonhuman   # Schotte alpha-1 = rat
    assert ("risperidone", "H1") in nonhuman       # Schotte H1 = guinea pig
    for rec in (r for r in schotte if not r.is_human):
        with pytest.raises(species_gate.UnsourcedParameterError):
            ot.gate_record(rec)


def test_risperidone_is_far_more_potent_than_haloperidol_at_5ht2a():
    """The Meltzer contrast the Siafis test rests on: risperidone's human 5-HT2A
    affinity is orders of magnitude higher than haloperidol's. This is read off
    the registered human values, not asserted."""
    halo = ot.human_ki_range_nM("haloperidol", "5HT2A")
    risp = ot.human_ki_range_nM("risperidone", "5HT2A")
    # The whole haloperidol human 5-HT2A range sits well above the whole
    # risperidone range: the two do not overlap.
    assert risp[1] < halo[0]
    assert halo[0] / risp[1] > 3


def test_neither_drug_has_meaningful_m1_affinity():
    """M1 is sourced but near the floor: every human value >= ~1.5 uM, and the
    ceiling rows are flagged censored. M1 cannot distinguish the two drugs' EPS."""
    for drug in ot.DRUGS:
        lo, hi = ot.human_ki_range_nM(drug, "M1")
        assert lo > 1000, "%s M1 min should be weak (>1 uM)" % drug
    ceilings = [r for r in ot.HUMAN_KI if r.target == "M1" and r.censored == ">"]
    assert {r.drug for r in ceilings} == set(ot.DRUGS)
    assert all(r.value_nM == 10000 for r in ceilings)


def test_human_ki_range_is_the_full_pdsp_span_including_extremes():
    """Each cell registers its PDSP minimum and maximum, so the reported range is
    the true span. Spot-check two cells against the download."""
    # haloperidol 5-HT2A human PDSP span 25 (Seeman 1997) - 435 (Fernandez 2005).
    assert ot.human_ki_range_nM("haloperidol", "5HT2A") == (25.0, 435.0)
    # risperidone 5-HT2A human PDSP span 0.14 (Seeman 1997) - 7 (Bymaster 1999).
    assert ot.human_ki_range_nM("risperidone", "5HT2A") == (0.14, 7.0)


def test_fingerprint_returns_all_five_targets_per_drug():
    for drug in ot.DRUGS:
        fp = ot.fingerprint(drug)
        assert set(fp) == set(ot.TARGETS)
        for lo, hi in fp.values():
            assert lo <= hi


def test_searched_and_empty_records_the_tissue_subclass_gaps():
    sought = {entry[0] for entry in ot.SEARCHED_AND_EMPTY}
    assert any("D3" in s and "tissue" in s for s in sought)
    assert any("M1" in s and "tissue" in s for s in sought)
    for _, outcome in ot.SEARCHED_AND_EMPTY:
        assert outcome
