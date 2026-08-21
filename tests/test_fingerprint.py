"""P8.1: the binding-fingerprint input API.

The load-bearing test in this file is the ROUND TRIP: a hand-built,
literature-calibrated scenario expressed as a PDSP-style Ki table, pushed back
through the fingerprint API, and required to reproduce the scenario it came
from. Two of them are exact and one is not, and the one that is not is
measured rather than excused; see `docs/architecture.md`, P8.1.
"""

import numpy as np
import pytest

from neurocascade import (
    acetylcholine,
    adenosine,
    cannabinoid,
    dopamine,
    fingerprint as fp,
    glutamate,
    histamine,
    monoamine_transporters,
    noradrenaline,
    opioid,
    scenarios,
    serotonin,
)
from neurocascade.legacy import (
    drug_pk,
)
from neurocascade.engine import Simulation
from neurocascade.receptor import Ligand, ReceptorSite, ligand_from_ki


# ---------------------------------------------------------------------------
# The conversion itself
# ---------------------------------------------------------------------------


def test_ligand_from_ki_reproduces_the_affinity_exactly():
    ligand = ligand_from_ki("x", ki_nM=3.6, kon=0.2, efficacy=1.0)
    assert ligand.kon == 0.2
    # 0.2 * 3.6 is 0.7200000000000001 in binary floating point, so this and
    # every "exact" claim below is exact to floating point and not bit-for-bit
    # against a decimal literal. `scenarios.py` writes the same koff as 0.72.
    assert ligand.koff == pytest.approx(0.72, rel=1e-15)
    assert ligand.koff / ligand.kon == pytest.approx(3.6)


def test_the_affinity_is_exact_for_every_kon_which_is_the_whole_point():
    """`koff/kon == Ki` whatever `kon` is: the assumption buys a timescale
    and costs nothing at equilibrium. This is the claim the module docstring
    and docs/architecture.md both make, asserted rather than repeated."""
    for kon in (1e-4, 0.01, 0.2, 5.82, 100.0):
        ligand = ligand_from_ki("x", ki_nM=17.3, kon=kon, efficacy=0.5)
        assert ligand.koff / ligand.kon == pytest.approx(17.3, rel=1e-12)


@pytest.mark.parametrize("ki", [0.0, -1.0, float("nan"), float("inf")])
def test_a_ki_that_is_not_a_measurement_is_refused(ki):
    with pytest.raises(ValueError):
        ligand_from_ki("x", ki_nM=ki, kon=0.2, efficacy=1.0)


@pytest.mark.parametrize("kon", [0.0, -1.0, float("nan")])
def test_an_unusable_kon_is_refused(kon):
    with pytest.raises(ValueError):
        ligand_from_ki("x", ki_nM=1.0, kon=kon, efficacy=1.0)


def test_the_four_modules_that_had_their_own_copy_now_share_this_one():
    """The refactor P8.1 did to `acetylcholine`, `glutamate`, `cannabinoid`
    and `opioid`: same numbers, one statement of the assumption. The opioid
    site's ligands are the check with teeth, because that module's `_ligand`
    also does two table lookups and its numbers are pinned elsewhere."""
    assert acetylcholine.ILLUSTRATIVE_KON == glutamate.ILLUSTRATIVE_KON
    assert glutamate.ILLUSTRATIVE_KON == cannabinoid.ILLUSTRATIVE_KON
    kon = cannabinoid.ILLUSTRATIVE_KON
    for ligand in cannabinoid.ligands_cb1():
        assert ligand.kon == kon
        assert ligand.koff == pytest.approx(kon * (ligand.koff / kon))
    for receptor, ligands in (("kappa", opioid.ligands_kappa()),
                              ("delta", opioid.ligands_delta())):
        site_kon = opioid.SITE_KON[receptor]
        for ligand in ligands:
            assert ligand.kon == site_kon
            assert ligand.koff == pytest.approx(
                site_kon * opioid.ki(ligand.name, receptor), rel=1e-12
            )


# ---------------------------------------------------------------------------
# The target registry
# ---------------------------------------------------------------------------


def test_every_target_reaches_a_real_system():
    for name, spec in fp.TARGETS.items():
        system = spec.system()
        assert system is not None
        if spec.is_transporter:
            assert system.transporter.n == len(system.pk_models)
        else:
            assert system.receptor_site.n == len(system.pk_models)


def test_every_kon_matches_the_module_it_came_from():
    """The registry restates no number. Each entry is checked against the
    module constant or the module's own ligand it was assembled from, so a
    module that changes its site kon and forgets this table fails here."""
    expected = {
        "MOR": opioid.KAPPA_KON_DIPRENORPHINE,
        "DOR": opioid.DELTA_KON_PCL_DPDPE,
        "KOR": opioid.KAPPA_KON_DIPRENORPHINE,
        "D1": dopamine.D1_KON_DOPAMINE,
        "5HT1A": serotonin.HT1A_KON,
        "5HT2A": serotonin.HT2A_KON,
        "alpha1": noradrenaline.ALPHA1_KON_PRAZOSIN,
        "alpha2": noradrenaline.ALPHA2_KON,
        "beta": noradrenaline.BETA_KON_PROPRANOLOL,
        "alpha4beta2": acetylcholine.ILLUSTRATIVE_KON,
        "alpha7": acetylcholine.ILLUSTRATIVE_KON,
        "M1": acetylcholine.ILLUSTRATIVE_KON,
        "M2": acetylcholine.ILLUSTRATIVE_KON,
        "NMDA": glutamate.ILLUSTRATIVE_KON,
        "AMPA": glutamate.ILLUSTRATIVE_KON,
        "H1": histamine.H1_KON_MEPYRAMINE,
        "CB1": cannabinoid.ILLUSTRATIVE_KON,
        "A2A": adenosine.A2A_KON_NECA,
    }
    for name, kon in expected.items():
        assert fp.TARGETS[name].kon == kon, name

    # The two read off a calibrated ligand rather than a constant.
    haloperidol = [
        l for l in scenarios.d2_dopamine_ligands_calibrated()
        if l.name == "haloperidol"
    ][0]
    assert fp.TARGETS["D2"].kon == haloperidol.kon
    diazepam = [
        l for l in scenarios.gaba_a_benzodiazepine_ligands_calibrated()
        if l.name == "diazepam"
    ][0]
    assert fp.TARGETS["GABA_A_BZD"].kon == diazepam.kon


def test_only_the_transporters_form_no_rate_pair():
    """`not-needed` is a statement about `transport.py` consuming a Ki
    directly, so it must apply to exactly the three transporters and to
    nothing else."""
    not_needed = {
        name for name, spec in fp.TARGETS.items()
        if spec.kon_provenance == "not-needed"
    }
    assert not_needed == set(monoamine_transporters.TRANSPORTERS)
    for name, spec in fp.TARGETS.items():
        assert spec.is_transporter == (spec.kon is None)


def test_the_illustrative_kon_targets_are_exactly_the_sites_with_no_kinetics():
    illustrative = {
        name for name, spec in fp.TARGETS.items()
        if spec.kon_provenance == "illustrative"
    }
    assert illustrative == {
        "alpha4beta2", "alpha7", "M1", "M2", "NMDA", "AMPA", "CB1"
    }


def test_mu_is_the_only_borrowed_kon_and_it_borrows_what_opioid_py_borrows():
    borrowed = {
        name for name, spec in fp.TARGETS.items()
        if spec.kon_provenance == "borrowed-from-subtype"
    }
    assert borrowed == {"MOR"}
    assert "mu" not in opioid.SITE_KON  # which is why the fallback is needed
    assert fp.TARGETS["MOR"].kon == opioid.KAPPA_KON_DIPRENORPHINE


def test_aliases_resolve_and_do_not_collide():
    assert fp.resolve_target("5-HT2A") == "5HT2A"
    assert fp.resolve_target("5ht2a") == "5HT2A"
    assert fp.resolve_target("MU") == "MOR"
    assert fp.resolve_target("OPRM1") == "MOR"
    assert fp.resolve_target("SLC6A4") == "SERT"
    assert fp.resolve_target("BZP Rat Brain Site") == "GABA_A_BZD"
    assert fp.resolve_target("Alpha-2") == "alpha2"


def test_an_unknown_target_is_refused_by_name_and_lists_what_exists():
    with pytest.raises(fp.UnknownTargetError) as caught:
        fp.resolve_target("sigma-1")
    assert "5HT2A" in str(caught.value)
    assert not fp.is_modelled("sigma-1")


def test_no_registry_system_binds_a_transmitter_pool():
    """`predict` runs the receptor layer with no circuit, so it has no
    transporter or projection pool for a `transmitter_sources` ligand to read.
    None of the registry's systems declares one, and `predict` raises rather
    than degrading if that ever changes."""
    for name, spec in fp.TARGETS.items():
        assert not getattr(spec.system(), "transmitter_sources", None), name


def test_every_target_joins_to_a_gene_the_expression_layer_carries():
    """P8.1 is the first place the pharmacological and anatomical name spaces
    meet, and the join has to land on genes that actually exist, because P8.2
    reads a density map off each one."""
    from neurocascade.legacy.expression import genes

    known = {gene.symbol for gene in genes.RECEPTOR_GENES}
    for name, spec in fp.TARGETS.items():
        assert spec.gene in known, (name, spec.gene)


def test_the_multi_subunit_targets_are_exactly_the_ones_flagged_as_proxies():
    """A single gene cannot be a pentamer or a tetramer, so every
    multi-subunit receptor's gene must carry a note saying so, and no
    single-gene receptor's may pretend to."""
    proxies = {
        name for name, spec in fp.TARGETS.items() if spec.gene_note
    }
    assert proxies == {
        "GABA_A_BZD", "alpha4beta2", "alpha7", "NMDA", "AMPA"
    }
    for name in proxies:
        assert "PROXY" in fp.TARGETS[name].gene_note


def test_the_target_report_covers_every_target():
    text = fp.target_report()
    for name in fp.TARGETS:
        assert name in text


# ---------------------------------------------------------------------------
# Building a fingerprint from PDSP-shaped rows
# ---------------------------------------------------------------------------


def _row(receptor, ki, ligand="testcompound", **kwargs):
    return fp.PDSPRow(receptor=receptor, test_ligand=ligand, ki_nM=ki, **kwargs)


def test_a_pdsp_table_becomes_a_fingerprint():
    rows = [
        _row("D2", 1.4, species="HUMAN", source="CLONED",
             hot_ligand="3H-SPIPERONE", reference="Roth 1994"),
        _row("5-HT2A", 12.0),
        _row("SERT", 300.0),
    ]
    print_ = fp.BindingFingerprint.from_pdsp_rows(
        rows, efficacy={"D2": 0.0, "5HT2A": 0.0, "SERT": 0.0}
    )
    assert print_.compound == "testcompound"
    assert print_.targets == ("5HT2A", "D2", "SERT")
    assert print_.ki("D2") == 1.4
    assert print_.entries["D2"].hot_ligand == "3H-SPIPERONE"
    assert print_.entries["D2"].reference == "Roth 1994"
    assert "5-HT2A" in print_


def test_rows_may_be_plain_mappings():
    rows = [{"receptor": "H1", "test_ligand": "x", "ki_nM": 5.0}]
    built = fp.BindingFingerprint.from_pdsp_rows(rows, efficacy={"H1": 0.0})
    assert built.ki("H1") == 5.0


def test_a_target_this_model_does_not_carry_is_recorded_not_dropped():
    rows = [_row("H1", 5.0), _row("sigma-1", 20.0), _row("5-HT7", 40.0)]
    built = fp.BindingFingerprint.from_pdsp_rows(rows, efficacy={"H1": 0.0})
    assert built.targets == ("H1",)
    assert set(built.unmodelled) == {"sigma-1", "5-HT7"}
    assert all(reason for reason in built.unmodelled.values())


def test_a_missing_efficacy_is_an_error_and_names_the_target():
    rows = [_row("D2", 1.4), _row("H1", 5.0)]
    with pytest.raises(fp.MissingEfficacyError) as caught:
        fp.BindingFingerprint.from_pdsp_rows(rows, efficacy={"D2": 0.0})
    assert "H1" in str(caught.value)


def test_efficacy_is_never_defaulted():
    """The single most important refusal in this module: a Ki table has no
    efficacy column, so nothing here may supply one."""
    rows = [_row("D2", 1.4)]
    with pytest.raises(fp.MissingEfficacyError):
        fp.BindingFingerprint.from_pdsp_rows(rows, efficacy={})
    with pytest.raises(fp.MissingEfficacyError):
        fp.BindingFingerprint.from_pdsp_rows(rows, efficacy=None)


def test_an_efficacy_with_no_affinity_behind_it_is_an_error():
    rows = [_row("D2", 1.4)]
    with pytest.raises(ValueError) as caught:
        fp.BindingFingerprint.from_pdsp_rows(
            rows, efficacy={"D2": 0.0, "H1": 1.0}
        )
    assert "H1" in str(caught.value)


def test_several_determinations_combine_by_geometric_mean_by_default():
    rows = [_row("D2", 1.0), _row("D2", 100.0)]
    built = fp.BindingFingerprint.from_pdsp_rows(rows, efficacy={"D2": 0.0})
    assert built.ki("D2") == pytest.approx(10.0)
    assert "2 determinations" in built.entries["D2"].note
    low = fp.BindingFingerprint.from_pdsp_rows(
        rows, efficacy={"D2": 0.0}, combine="min"
    )
    high = fp.BindingFingerprint.from_pdsp_rows(
        rows, efficacy={"D2": 0.0}, combine="max"
    )
    assert (low.ki("D2"), high.ki("D2")) == (1.0, 100.0)


def test_a_single_determination_is_not_perturbed_by_the_averaging():
    """`exp(mean(log(x)))` of one value differs from it in the last bit, and
    an exact round trip cannot afford that."""
    built = fp.BindingFingerprint.from_pdsp_rows(
        [_row("SERT", 3.0)], efficacy={"SERT": 0.0}
    )
    assert built.ki("SERT") == 3.0


def test_two_aliases_of_one_target_may_not_carry_different_efficacies():
    """`{"5-HT2A": 0.0, "HTR2A": 1.0}` is a caller who thinks they described
    two receptors. Collapsing it would turn an antagonist into a full agonist
    without a word, so it is refused."""
    rows = [_row("5-HT2A", 10.0), _row("HTR2A", 1000.0)]
    with pytest.raises(ValueError) as caught:
        fp.BindingFingerprint.from_pdsp_rows(
            rows, efficacy={"5-HT2A": 0.0, "HTR2A": 1.0}
        )
    assert "5HT2A" in str(caught.value)
    # The same value under two aliases is fine, and the Ki values still merge.
    built = fp.BindingFingerprint.from_pdsp_rows(
        rows, efficacy={"5-HT2A": 0.0, "HTR2A": 0.0}
    )
    assert built.ki("5HT2A") == pytest.approx(100.0)


def test_a_merged_ki_does_not_claim_one_rows_provenance():
    """A Ki averaged over three preparations belongs to none of them."""
    rows = [
        _row("D2", 1.0, species="HUMAN", source="CLONED", reference="A"),
        _row("D2", 100.0, species="RAT", source="STRIATUM", reference="B"),
    ]
    built = fp.BindingFingerprint.from_pdsp_rows(rows, efficacy={"D2": 0.0})
    entry = built.entries["D2"]
    assert entry.species is None
    assert entry.source is None
    assert entry.reference is None
    assert "species differs across them (HUMAN, RAT)" in entry.note
    # Unanimous fields survive.
    agreeing = fp.BindingFingerprint.from_pdsp_rows(
        [_row("D2", 1.0, species="RAT"), _row("D2", 100.0, species="RAT")],
        efficacy={"D2": 0.0},
    )
    assert agreeing.entries["D2"].species == "RAT"


def test_a_table_that_covers_nothing_modelled_is_refused():
    """An empty fingerprint would predict nothing while looking like a
    prediction, so it is an error rather than a zero-target object."""
    with pytest.raises(fp.UnknownTargetError) as caught:
        fp.BindingFingerprint.from_pdsp_rows(
            [_row("sigma-1", 1.0), _row("5-HT7", 2.0)], efficacy={}
        )
    assert "sigma-1" in str(caught.value)


def test_an_unknown_combine_rule_is_refused_even_with_nothing_to_combine():
    with pytest.raises(ValueError):
        fp.BindingFingerprint.from_pdsp_rows(
            [_row("D2", 1.0)], efficacy={"D2": 0.0}, combine="nonsense"
        )


def test_a_table_covering_two_compounds_is_refused():
    rows = [_row("D2", 1.0, ligand="a"), _row("H1", 2.0, ligand="b")]
    with pytest.raises(ValueError):
        fp.BindingFingerprint.from_pdsp_rows(
            rows, efficacy={"D2": 0.0, "H1": 0.0}
        )


# ---------------------------------------------------------------------------
# Ligand-set construction
# ---------------------------------------------------------------------------


def test_a_ligand_set_spans_every_target_the_compound_binds():
    rows = [_row("MU", 1.0), _row("kappa", 50.0), _row("DAT", 400.0)]
    built = fp.BindingFingerprint.from_pdsp_rows(
        rows, efficacy={"MOR": 1.0, "KOR": 0.3, "DAT": 0.0}
    )
    ligands = fp.ligand_set(built)
    assert ligands.targets == ("DAT", "KOR", "MOR")
    assert ligands["MOR"].koff / ligands["MOR"].kon == pytest.approx(1.0)
    assert ligands["KOR"].efficacy == 0.3
    assert ligands["DAT"].ki == 400.0
    assert ligands["DAT"].is_blocker
    assert ligands.provenance["DAT"] == (None, "not-needed")
    assert ligands.provenance["MOR"] == (
        opioid.KAPPA_KON_DIPRENORPHINE, "borrowed-from-subtype"
    )


def test_a_transporter_entry_is_a_pure_blocker_unless_told_otherwise():
    """A Ki table cannot say a compound is a substrate; that is a release
    assay. So the default is a blocker and the caller has to supply the
    release parameters to get anything else."""
    built = fp.BindingFingerprint.from_pdsp_rows(
        [_row("DAT", 560.0)], efficacy={"DAT": 0.0}
    )
    assert fp.ligand_set(built)["DAT"].is_blocker

    built.entries["DAT"].release_ec50 = 6.4
    built.entries["DAT"].release_emax = 1.03
    assert fp.ligand_set(built)["DAT"].is_substrate


def test_restricting_to_a_target_the_compound_does_not_bind_is_an_error():
    built = fp.BindingFingerprint.from_pdsp_rows(
        [_row("H1", 5.0)], efficacy={"H1": 0.0}
    )
    with pytest.raises(KeyError):
        fp.ligand_set(built, targets=["D2"])


def test_the_channel_pool_is_refused_where_a_site_has_none():
    built = fp.BindingFingerprint.from_pdsp_rows(
        [_row("H1", 5.0)], efficacy={"H1": 0.0}, pools={"H1": "channel"}
    )
    with pytest.raises(ValueError):
        fp.ligand_set(built)


def test_a_channel_pool_entry_at_nmda_takes_the_measured_rate():
    """The one place a fingerprint ligand gets a MEASURED kon rather than an
    illustrative one, because [Dravid2007] measured MK-801's, and the only
    place the reported provenance differs from the target's own."""
    built = fp.BindingFingerprint.from_pdsp_rows(
        [_row("NMDA", 15.0)], efficacy={"NMDA": 0.0}, pools={"NMDA": "channel"}
    )
    ligands = fp.ligand_set(built)
    assert ligands["NMDA"].kon == glutamate.NMDA_CHANNEL_KON
    assert ligands["NMDA"].kon != glutamate.ILLUSTRATIVE_KON
    assert ligands.provenance["NMDA"] == (
        glutamate.NMDA_CHANNEL_KON, "measured-at-site"
    )
    assert fp.TARGETS["NMDA"].kon_provenance == "illustrative"


def test_the_channel_pool_targets_are_the_sites_that_actually_have_one():
    """`DesensitizingReceptorSite` carries a channel-blocker pool and
    deliberately does not subclass `ChannelBlockingReceptorSite`, so the two
    nicotinic sites have one and an isinstance check would miss them."""
    with_pool = set()
    for name, spec in fp.TARGETS.items():
        if spec.is_transporter:
            continue
        site = spec.system().receptor_site
        if hasattr(site, "channel_blockers"):
            with_pool.add(name)
    assert with_pool == set(fp._CHANNEL_KON)
    assert with_pool == {"NMDA", "AMPA", "alpha4beta2", "alpha7"}


def test_a_channel_ligand_at_a_nicotinic_site_is_accepted_and_labelled():
    built = fp.BindingFingerprint.from_pdsp_rows(
        [_row("alpha4beta2", 3000.0)], efficacy={"alpha4beta2": 0.0},
        pools={"alpha4beta2": "channel"},
    )
    ligands = fp.ligand_set(built)
    assert ligands.provenance["alpha4beta2"] == (
        acetylcholine.ILLUSTRATIVE_KON, "illustrative"
    )
    system = fp.receptor_systems_for(built, _bolus_pk())["alpha4beta2"]
    site = system.receptor_site
    base = acetylcholine.receptor_system_alpha4beta2().receptor_site
    assert site.n_blockers == base.n_blockers + 1
    assert site.n_orthosteric == base.n_orthosteric


# ---------------------------------------------------------------------------
# Wiring a fingerprint ligand into an existing system
# ---------------------------------------------------------------------------


def _bolus_pk():
    from neurocascade.pk import OneCompartmentPK

    return OneCompartmentPK(ke=0.05)


def test_adding_a_ligand_leaves_the_system_otherwise_untouched():
    base = scenarios.gaba_a_receptor_system_calibrated()
    ligand = ligand_from_ki("newcompound", ki_nM=10.0, kon=0.2, efficacy=0.4)
    widened = fp.add_to_system(base, ligand, _bolus_pk())

    assert widened.coupling == base.coupling
    assert widened.connection_modulation == base.connection_modulation
    assert widened.name == base.name
    assert widened.receptor_site.name == base.receptor_site.name
    assert [l.name for l in widened.receptor_site.ligands] == [
        l.name for l in base.receptor_site.ligands
    ] + ["newcompound"]
    for before, after in zip(base.receptor_site.ligands,
                             widened.receptor_site.ligands):
        assert (after.kon, after.koff, after.efficacy) == (
            before.kon, before.koff, before.efficacy
        )
    assert len(widened.pk_models) == widened.receptor_site.n


def test_a_duplicate_ligand_is_refused_by_default():
    base = scenarios.gaba_a_receptor_system_calibrated()
    ligand = ligand_from_ki("diazepam", ki_nM=3.6, kon=0.2, efficacy=1.0)
    with pytest.raises(ValueError) as caught:
        fp.add_to_system(base, ligand, _bolus_pk())
    assert "on_duplicate" in str(caught.value)


def test_a_replacement_keeps_the_ligand_in_its_original_position():
    base = scenarios.gaba_a_receptor_system_calibrated()
    ligand = ligand_from_ki("diazepam", ki_nM=3.6, kon=0.2, efficacy=1.0)
    widened = fp.add_to_system(
        base, ligand, _bolus_pk(), on_duplicate="replace"
    )
    assert [l.name for l in widened.receptor_site.ligands] == [
        l.name for l in base.receptor_site.ligands
    ]
    assert widened.receptor_site.ligands[0] is ligand


def test_a_desensitizing_site_gains_a_ligand_that_does_not_desensitize():
    """A Ki says nothing about desensitization, so k_des is 0 and the added
    ligand's desensitized state stays at zero for all time."""
    base = acetylcholine.receptor_system_alpha4beta2()
    site = base.receptor_site
    ligand = ligand_from_ki("newcompound", ki_nM=100.0,
                            kon=acetylcholine.ILLUSTRATIVE_KON, efficacy=0.0)
    widened = fp.add_to_system(base, ligand, _bolus_pk())
    new_site = widened.receptor_site

    assert new_site.n_orthosteric == site.n_orthosteric + 1
    assert new_site.n_blockers == site.n_blockers
    assert new_site.k_des[-1] == 0.0
    assert list(new_site.k_des[:-1]) == list(site.k_des)
    assert list(new_site.k_rec[:-1]) == list(site.k_rec)
    # The blockers keep their own pool, so the ligand order is
    # orthosteric-then-blocker and the PK models follow it.
    assert [l.name for l in new_site.ligands][-1] == "mecamylamine"
    assert len(widened.pk_models) == new_site.n


def test_a_desensitized_ligand_still_counts_as_occupying_the_pool():
    """`DesensitizingReceptorSite` holds a ligand in TWO states and its own
    docstring says the desensitized one "occupies the pool (so it still
    competes)". Reporting only the conducting state would understate what
    fraction of the receptor the drug holds, by up to `1 + k_des/k_rec`."""
    entry = fp.FingerprintEntry(
        target="alpha4beta2", ki_nM=20.0, efficacy=1.0, k_des=0.3, k_rec=0.02
    )
    built = fp.BindingFingerprint(
        compound="diazepam", entries={"alpha4beta2": entry}
    )
    prediction = fp.predict(
        built, dose=5.0, route="IP", species="rat", body_weight_kg=0.25,
        t_span=(0.0, 60.0), n_times=61,
    )
    at_site = prediction["alpha4beta2"]
    system = at_site.system
    site = system.receptor_site
    index = [l.name for l in site.ligands].index("diazepam")

    # The reported occupancy is bound + desensitized, and the conducting
    # state alone is much smaller.
    states = fp._integrate_block(
        system.pk_models,
        [5.0 if l.name == "diazepam" else 0.0 for l in site.ligands],
        site.n_states, site.occupancy_rhs, np.zeros(site.n_states),
        (0.0, 60.0), np.linspace(0.0, 60.0, 61), "LSODA",
    )["block"]
    bound = states[index]
    desensitized = states[site.n_orthosteric + index]
    assert np.allclose(at_site.occupancy, bound + desensitized)
    assert at_site.peak_occupancy > 1.5 * float(bound.max())
    # Activation, by contrast, counts only the conducting fraction.
    assert float(at_site.activation.max()) <= float(bound.max()) + 1e-12


def test_a_blockers_occupancy_is_read_from_the_right_state():
    """At a desensitizing site the ligand order is orthosteric-then-blocker
    and the state order is bound, desensitized, blocked, so blocker `j` is
    ligand `p + j` and state `2p + j`. Indexing the state block by the ligand
    index would return an unrelated ligand's desensitized state."""
    site = acetylcholine.receptor_system_alpha4beta2().receptor_site
    p, b = site.n_orthosteric, site.n_blockers
    states = np.zeros((site.n_states, 1))
    states[2 * p, 0] = 0.7  # the first blocker's blocked state
    states[p, 0] = 0.4      # the first orthosteric ligand's desensitized state
    assert b >= 1
    assert fp.ligand_occupancy(site, states, p)[0] == 0.7
    assert fp.ligand_occupancy(site, states, 0)[0] == 0.4


def test_a_replacement_may_not_silently_keep_the_old_desensitization_rates():
    base = acetylcholine.receptor_system_alpha4beta2()
    site = base.receptor_site
    index = [l.name for l in site.orthosteric_ligands].index("nicotine")
    assert site.k_des[index] > 0.0

    replacement = ligand_from_ki(
        "nicotine", ki_nM=10.0, kon=acetylcholine.ILLUSTRATIVE_KON,
        efficacy=1.0,
    )
    stated = fp.add_to_system(
        base, replacement, _bolus_pk(), k_des=5.0, k_rec=1.0,
        on_duplicate="replace",
    )
    assert stated.receptor_site.k_des[index] == 5.0
    assert stated.receptor_site.k_rec[index] == 1.0

    # Saying nothing keeps the module's own measured rates, which is the
    # other half of "not silently".
    silent = fp.add_to_system(
        base, replacement, _bolus_pk(), on_duplicate="replace"
    )
    assert silent.receptor_site.k_des[index] == site.k_des[index]


def test_a_channel_blocking_site_accepts_a_declared_channel_ligand():
    base = glutamate.receptor_system_nmda()
    site = base.receptor_site
    ligand = ligand_from_ki("newblocker", ki_nM=50.0,
                            kon=glutamate.NMDA_CHANNEL_KON, efficacy=0.0)
    widened = fp.add_to_system(base, ligand, _bolus_pk(), pool="channel")
    assert widened.receptor_site.n_blockers == site.n_blockers + 1
    assert widened.receptor_site.n_orthosteric == site.n_orthosteric
    assert widened.receptor_site.synaptic_share == site.synaptic_share


def test_a_transporter_system_gains_a_drug_and_keeps_its_kinetics():
    base = monoamine_transporters.transporter_system("DAT")
    ligand = fp.TransporterLigand(name="newcompound", ki=250.0)
    widened = fp.add_to_transporter_system(base, ligand, _bolus_pk())
    for attribute in ("km", "vmax", "clearance_rate", "basal", "release_rate",
                      "release_capacity", "transmitter", "name"):
        assert getattr(widened.transporter, attribute) == getattr(
            base.transporter, attribute
        )
    assert widened.transporter.n == base.transporter.n + 1


def test_receptor_systems_for_spans_every_target():
    rows = [_row("D2", 1.4), _row("H1", 5.0), _row("SERT", 300.0)]
    built = fp.BindingFingerprint.from_pdsp_rows(
        rows, efficacy={"D2": 0.0, "H1": 0.0, "SERT": 0.0}
    )
    systems = fp.receptor_systems_for(built, _bolus_pk())
    assert set(systems) == {"D2", "H1", "SERT"}
    assert "testcompound" in [l.name for l in systems["D2"].receptor_site.ligands]
    assert "testcompound" in [l.name for l in systems["SERT"].transporter.ligands]
    # Every one of them is something `engine.Simulation` will accept.
    assert systems["D2"].coupling == dopamine.receptor_system_d2().coupling


# ---------------------------------------------------------------------------
# THE ROUND TRIP
# ---------------------------------------------------------------------------


def _gaba_a_fingerprints():
    """Diazepam and flumazenil as PDSP-shaped tables, from the two papers
    `scenarios.py`'s calibrated block already cites."""
    diazepam = fp.BindingFingerprint.from_pdsp_rows(
        [fp.PDSPRow(
            receptor="BZP Rat Brain Site", test_ligand="diazepam", ki_nM=3.6,
            species="RAT", source="CEREBRAL CORTEX", hot_ligand="3H-DIAZEPAM",
            reference="Mohler & Okada, Life Sci 1977;20:2101",
        )],
        efficacy={"GABA_A_BZD": (1.0, "potentiation strength; see scenarios.py")},
    )
    flumazenil = fp.BindingFingerprint.from_pdsp_rows(
        [fp.PDSPRow(
            receptor="BZP Rat Brain Site", test_ligand="flumazenil", ki_nM=1.0,
            species="RAT", source="CEREBRAL CORTEX",
            hot_ligand="3H-RO 15-1788",
            reference="Mohler et al., J Neurochem 1981;37:714",
        )],
        efficacy={"GABA_A_BZD": (0.0, "neutral competitive antagonist")},
    )
    return diazepam, flumazenil


def test_round_trip_the_gaba_a_ligands_are_reproduced_exactly():
    """The published Ki, back through the API, IS the hand-built ligand.

    Exact because `scenarios.gaba_a_benzodiazepine_ligands_calibrated` built
    both ligands the way this module does: one shared kon and
    `koff = kon * Ki`.
    """
    diazepam, flumazenil = _gaba_a_fingerprints()
    rebuilt = {
        "diazepam": fp.ligand_set(diazepam)["GABA_A_BZD"],
        "flumazenil": fp.ligand_set(flumazenil)["GABA_A_BZD"],
    }
    for hand_built in scenarios.gaba_a_benzodiazepine_ligands_calibrated():
        got = rebuilt[hand_built.name]
        assert got.kon == hand_built.kon
        assert got.koff == pytest.approx(hand_built.koff, rel=1e-12)
        assert got.efficacy == hand_built.efficacy


def test_round_trip_the_gaba_a_scenario_runs_identically():
    """The whole hand-built scenario, with its receptor site rebuilt from the
    two Ki values through the fingerprint API: same circuit, same PK, same
    trajectories to integrator noise."""
    diazepam, flumazenil = _gaba_a_fingerprints()
    network = scenarios.amygdala_anxiety_circuit_network()

    hand_built = scenarios.gaba_a_receptor_system_calibrated()
    system = hand_built
    for print_ in (diazepam, flumazenil):
        system = fp.add_to_system(
            system,
            fp.ligand_set(print_)["GABA_A_BZD"],
            hand_built.pk_models[
                [l.name for l in hand_built.receptor_site.ligands].index(
                    print_.compound
                )
            ],
            on_duplicate="replace",
        )

    assert [l.name for l in system.receptor_site.ligands] == [
        l.name for l in hand_built.receptor_site.ligands
    ]

    t_eval = np.linspace(0.0, 40.0, 81)
    doses = [100.0, 0.0]
    reference = Simulation(
        network=network, receptor_systems=[hand_built]
    ).run(doses=doses, t_span=(0.0, 40.0), t_eval=t_eval)
    through_api = Simulation(
        network=scenarios.amygdala_anxiety_circuit_network(),
        receptor_systems=[system],
    ).run(doses=doses, t_span=(0.0, 40.0), t_eval=t_eval)

    assert np.allclose(through_api["occupancy"], reference["occupancy"],
                       rtol=0, atol=1e-12)
    assert np.allclose(through_api["rates"], reference["rates"],
                       rtol=0, atol=1e-12)


def test_round_trip_the_kappa_and_delta_ligands_are_reproduced_exactly():
    """P5.7's kappa and delta sites are derivation 3 over `opioid.KI_HUMAN_CHO`
    with a site kon, which is exactly what this module does, so every ligand
    at both sites round-trips bit for bit."""
    for target, receptor, hand_built in (
        ("KOR", "kappa", opioid.ligands_kappa()),
        ("DOR", "delta", opioid.ligands_delta()),
    ):
        for ligand in hand_built:
            name = ligand.name
            built = fp.BindingFingerprint.from_pdsp_rows(
                [fp.PDSPRow(receptor=receptor, test_ligand=name,
                            ki_nM=opioid.ki(name, receptor))],
                efficacy={target: opioid.efficacy(name, receptor)},
            )
            got = fp.ligand_set(built)[target]
            assert got.kon == ligand.kon, (target, name)
            assert got.koff == pytest.approx(ligand.koff, rel=1e-12), (target, name)
            assert got.efficacy == ligand.efficacy, (target, name)


def test_round_trip_the_mu_ligands_reproduce_the_affinity_and_not_the_rates():
    """The round trip that is NOT exact, reported rather than hidden.

    `scenarios.mu_opioid_ligands_calibrated` used DERIVATION 2: fentanyl's and
    naloxone's kon each come from that ligand's own measured koff [Mann2022]
    divided by its own Ki. A fingerprint has only the Ki, so it uses the
    site's shared kon, and the two agree on affinity and disagree on rate.
    The disagreement is measured here so that its size is a number in the
    test suite rather than a claim in a docstring.
    """
    published_ki = {"fentanyl": 1.346, "naloxone": 1.518}
    hand_built = {l.name: l for l in scenarios.mu_opioid_ligands_calibrated()}
    efficacies = {"fentanyl": 1.0, "naloxone": 0.0}

    ratios = {}
    for name, ki in published_ki.items():
        built = fp.BindingFingerprint.from_pdsp_rows(
            [fp.PDSPRow(receptor="MU", test_ligand=name, ki_nM=ki)],
            efficacy={"MOR": efficacies[name]},
        )
        got = fp.ligand_set(built)["MOR"]
        reference = hand_built[name]

        # The affinity agrees to the rounding of the hand-built literals.
        assert got.koff / got.kon == pytest.approx(
            reference.koff / reference.kon, rel=2e-4
        )
        assert got.efficacy == reference.efficacy
        ratios[name] = got.koff / reference.koff

    # And the rates do not: naloxone is the one that matters, because its
    # measured residence time at mu is what drives reversal kinetics.
    assert ratios["fentanyl"] == pytest.approx(0.7457, rel=1e-3)
    assert ratios["naloxone"] == pytest.approx(0.09203, rel=1e-3)


def test_round_trip_the_d2_ligands_reproduce_the_affinity_and_not_the_rates():
    """The same non-exactness at D2, and for a sharper reason: BOTH rate
    constants were measured there ([Agren2021], [Sykes2017]), so derivation 3
    is strictly less informed than the hand-built pair and the site kon it
    borrows is haloperidol's."""
    hand_built = {l.name: l for l in scenarios.d2_dopamine_ligands_calibrated()}
    for name, efficacy in (("dopamine", 1.0), ("haloperidol", 0.0)):
        reference = hand_built[name]
        ki = reference.koff / reference.kon
        built = fp.BindingFingerprint.from_pdsp_rows(
            [fp.PDSPRow(receptor="D2", test_ligand=name, ki_nM=ki)],
            efficacy={"D2": efficacy},
        )
        got = fp.ligand_set(built)["D2"]
        assert got.koff / got.kon == pytest.approx(ki, rel=1e-12)
        if name == "haloperidol":
            # The site kon IS haloperidol's, so this one is exact.
            assert got.kon == reference.kon
            assert got.koff == pytest.approx(reference.koff, rel=1e-12)
        else:
            assert got.kon != reference.kon


# ---------------------------------------------------------------------------
# The standalone integrator, and predict()
# ---------------------------------------------------------------------------


def test_the_circuit_free_rhs_is_the_engines_rhs():
    """`_CircuitFreeBlock` is a second implementation of a right-hand side the
    engine already has, so the two are compared where the comparison can be
    EXACT: at the derivative, at the same state.

    Comparing trajectories instead would only show that two `solve_ivp` calls
    over two different-length state vectors landed close, which is a statement
    about the integrator rather than about the model.
    """
    hand_built = scenarios.gaba_a_receptor_system_calibrated()
    site = hand_built.receptor_site
    simulation = Simulation(
        network=scenarios.amygdala_anxiety_circuit_network(),
        receptor_systems=[hand_built],
    )
    block = fp._CircuitFreeBlock(
        hand_built.pk_models, site.n_states, site.occupancy_rhs
    )

    rng = np.random.default_rng(20260814)
    for _ in range(8):
        pk_state = rng.uniform(0.0, 200.0, simulation.d)
        occupancy = rng.uniform(0.0, 0.4, simulation.r)
        rates = rng.uniform(0.0, 1.0, simulation.c)
        y_engine = np.concatenate([pk_state, occupancy, rates])
        y_block = np.concatenate([pk_state, occupancy])

        engine_derivative = simulation.rhs(0.0, y_engine)
        block_derivative = block.rhs(0.0, y_block)

        # The PK region and the occupancy region, bit for bit.
        assert np.array_equal(
            block_derivative, engine_derivative[: simulation.d + simulation.r]
        )


def test_the_circuit_free_integration_tracks_the_engines():
    """And the trajectories do land together, to the integrators' own error.

    Both are RK45 with `max_step=0.1` but over state vectors of different
    length, so `solve_ivp` picks different steps and the two disagree by its
    error control rather than by anything in the model: the largest occupancy
    difference over 40 minutes is 1.8e-05, against occupancies of order 0.7.
    The exactness claim is the test above; this one bounds what the
    integrator adds on top of it.
    """
    hand_built = scenarios.gaba_a_receptor_system_calibrated()
    site = hand_built.receptor_site
    t_eval = np.linspace(0.0, 40.0, 81)
    doses = [100.0, 25.0]

    reference = Simulation(
        network=scenarios.amygdala_anxiety_circuit_network(),
        receptor_systems=[hand_built],
    ).run(doses=doses, t_span=(0.0, 40.0), t_eval=t_eval)

    standalone = fp._integrate_block(
        hand_built.pk_models, doses, site.n_states, site.occupancy_rhs,
        np.zeros(site.n_states), (0.0, 40.0), t_eval, "RK45",
    )
    worst = np.max(np.abs(standalone["block"] - reference["occupancy"]))
    assert worst < 5e-5
    assert np.allclose(
        standalone["receptor_concentrations"],
        reference["receptor_concentrations"],
        atol=1e-9,
    )


def test_predict_takes_a_fingerprint_a_dose_a_route_and_a_species():
    diazepam, _ = _gaba_a_fingerprints()
    prediction = fp.predict(
        diazepam, dose=2.0, route="IP", species="rat", body_weight_kg=0.25,
        on_duplicate="replace", t_span=(0.0, 60.0), n_times=61,
    )
    assert prediction.compound == "diazepam"
    assert prediction.route == "IP"
    assert prediction.brain_partition_status[0] == "receptor-drivable"

    at_site = prediction["GABA_A_BZD"]
    assert at_site.compartment == "unbound brain"
    assert 0.0 < at_site.peak_occupancy < 1.0
    assert at_site.occupancy[0] == pytest.approx(0.0)
    assert at_site.time_of_peak_occupancy_min > 0.0
    # The activation is the site's, so with only diazepam dosed it equals the
    # diazepam occupancy times its efficacy of 1.0.
    assert np.allclose(at_site.activation, at_site.occupancy)


def test_predict_reproduces_the_hand_built_dose_response_at_the_same_dose():
    """The round trip carried all the way through `predict`: the fingerprint
    path and the hand-built path, given the same dose, route and species,
    reach the same occupancy."""
    diazepam, _ = _gaba_a_fingerprints()
    t_eval = np.linspace(0.0, 60.0, 61)

    prediction = fp.predict(
        diazepam, dose=2.0, route="IP", species="rat", body_weight_kg=0.25,
        pk_model=drug_pk.diazepam_rat_ip_pk(body_weight_kg=0.25),
        on_duplicate="replace", t_span=(0.0, 60.0), n_times=61,
    )

    hand_built = scenarios.gaba_a_receptor_system_calibrated()
    site = hand_built.receptor_site
    reference = fp._integrate_block(
        [drug_pk.diazepam_rat_ip_pk(body_weight_kg=0.25), hand_built.pk_models[1]],
        [2.0, 0.0], site.n_states, site.occupancy_rhs,
        np.zeros(site.n_states), (0.0, 60.0), t_eval, "LSODA",
    )
    assert np.allclose(
        prediction["GABA_A_BZD"].occupancy, reference["block"][0], atol=1e-9
    )


def test_predict_refuses_a_compound_with_no_sourced_pharmacokinetics():
    """A binding fingerprint is not enough to predict from a DOSE: the other
    half is disposition, and this codebase has parameter sets for three
    drugs. The API says so instead of substituting one."""
    built = fp.BindingFingerprint.from_pdsp_rows(
        [_row("D2", 1.4, ligand="clozapine")], efficacy={"D2": 0.0}
    )
    with pytest.raises(drug_pk.UnsourcedParameterError) as caught:
        fp.predict(built, dose=1.0, route="IP", species="rat",
                   body_weight_kg=0.25)
    assert "clozapine" in str(caught.value)


def test_predict_refuses_a_route_with_no_sourced_absorption():
    diazepam, _ = _gaba_a_fingerprints()
    with pytest.raises(drug_pk.UnsourcedParameterError):
        fp.predict(diazepam, dose=2.0, route="SC", species="rat",
                   body_weight_kg=0.25, on_duplicate="replace")


def test_predict_with_a_caller_supplied_pk_model_needs_no_registry_entry():
    """The escape hatch, and the only one: a caller who owns a compound's PK
    provenance may pass the model in."""
    from neurocascade.pk import OneCompartmentPK

    built = fp.BindingFingerprint.from_pdsp_rows(
        [_row("H1", 5.0, ligand="unknowncompound")], efficacy={"H1": 0.0}
    )
    prediction = fp.predict(
        built, dose=100.0, route="IV", species="rat", body_weight_kg=0.25,
        pk_model=OneCompartmentPK(ke=0.05), t_span=(0.0, 30.0), n_times=31,
    )
    assert prediction["H1"].peak_occupancy > 0.0
    assert prediction.brain_partition_status[0] == "caller-supplied"


def test_the_reported_partition_status_describes_the_model_that_was_used():
    """`drug_pk`'s record is about a compound; a supplied PK model is about
    this run. For the arbitrary compound this API exists to accept the two
    always disagree, because the lookup is 'absent' for every unmodelled name
    while the supplied model may carry a partition. The run wins.
    """
    built = fp.BindingFingerprint.from_pdsp_rows(
        [_row("MOR", 20.0, ligand="novelcompound")], efficacy={"MOR": 1.0}
    )
    prediction = fp.predict(
        built, dose=5.0, route="IP", species="rat", body_weight_kg=0.25,
        pk_model=drug_pk.diazepam_rat_ip_pk(body_weight_kg=0.25),
        t_span=(0.0, 30.0), n_times=31,
    )
    # The lookup for this compound says "absent", and would have been a lie.
    assert drug_pk.brain_partition_status("novelcompound", "rat")[0] == "absent"
    assert prediction.brain_partition_status[0] == "receptor-drivable"
    assert prediction["MOR"].compartment == "unbound brain"


def test_predict_at_a_transporter_reports_the_transmitter_not_an_occupancy():
    from neurocascade.pk import OneCompartmentPK

    built = fp.BindingFingerprint.from_pdsp_rows(
        [_row("SERT", 3.0, ligand="unknowncompound")], efficacy={"SERT": 0.0}
    )
    prediction = fp.predict(
        built, dose=500.0, route="IV", species="rat", body_weight_kg=0.25,
        pk_model=OneCompartmentPK(ke=0.02), t_span=(0.0, 60.0), n_times=61,
    )
    at_sert = prediction["SERT"]
    assert at_sert.occupancy is None
    assert at_sert.transmitter is not None
    # Blocking uptake raises the extracellular transmitter above basal.
    basal = monoamine_transporters.transporter_system("SERT").transporter.basal
    assert at_sert.transmitter[0] == pytest.approx(basal)
    assert at_sert.transmitter.max() > basal


def test_a_zero_dose_leaves_every_target_untouched():
    diazepam, _ = _gaba_a_fingerprints()
    prediction = fp.predict(
        diazepam, dose=0.0, route="IP", species="rat", body_weight_kg=0.25,
        on_duplicate="replace", t_span=(0.0, 30.0), n_times=31,
    )
    assert np.allclose(prediction["GABA_A_BZD"].occupancy, 0.0)


def test_the_occupancy_ranking_excludes_transporters_and_sorts_by_peak():
    from neurocascade.pk import OneCompartmentPK

    rows = [_row("H1", 5.0, ligand="x"), _row("D2", 500.0, ligand="x"),
            _row("SERT", 3.0, ligand="x")]
    built = fp.BindingFingerprint.from_pdsp_rows(
        rows, efficacy={"H1": 0.0, "D2": 0.0, "SERT": 0.0}
    )
    prediction = fp.predict(
        built, dose=50.0, route="IV", species="rat", body_weight_kg=0.25,
        pk_model=OneCompartmentPK(ke=0.02), t_span=(0.0, 30.0), n_times=31,
    )
    ranking = prediction.occupancy_ranking()
    assert [name for name, _ in ranking] == ["H1", "D2"]
    assert ranking[0][1] > ranking[1][1]


# ---------------------------------------------------------------------------
# The PK dispatcher P8.1 added
# ---------------------------------------------------------------------------


def test_the_modelled_route_table_matches_the_builders():
    """`drug_pk.MODELLED_ROUTES` is written down rather than derived, so it is
    checked here, where a body weight may be supplied, against what the
    builders actually do for every pair and every route."""
    from neurocascade.pk import ROUTES

    weights = {"rat": 0.25, "mouse": 0.025}
    for pair in drug_pk.PK_BUILDERS:
        drug, species = pair
        works = []
        for route in ROUTES:
            try:
                drug_pk.pk_model_for(
                    drug, species, route, body_weight_kg=weights[species]
                )
            except drug_pk.UnsourcedParameterError:
                continue
            works.append(route)
        assert tuple(works) == drug_pk.MODELLED_ROUTES.get(pair, ()), pair


def test_the_dispatcher_names_what_it_does_have():
    with pytest.raises(drug_pk.UnsourcedParameterError) as caught:
        drug_pk.pk_model_for("morphine", "rat", "IV", body_weight_kg=0.25)
    message = str(caught.value)
    assert "diazepam/rat" in message and "fentanyl/rat" in message


def test_the_dispatcher_returns_the_same_model_as_the_named_builder():
    direct = drug_pk.diazepam_rat_ip_pk(body_weight_kg=0.25)
    dispatched = drug_pk.pk_model_for(
        "diazepam", "rat", "IP", body_weight_kg=0.25
    )
    assert dispatched.terminal_half_life_min == direct.terminal_half_life_min
    assert np.allclose(dispatched.initial_state(2.0), direct.initial_state(2.0))
