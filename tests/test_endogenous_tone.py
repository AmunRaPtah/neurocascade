"""P6.2: endogenous transmitter tone.

Four things are tested, in the order the acceptance criteria name them:

1. every projection system has a CITED basal concentration, with its
   provenance attached and no unit scale left anywhere;
2. an administered drug COMPETES with that tone in one shared pool, which is
   asserted against the integrated ODE rather than against the algebra the
   dose-ratio formula came from;
3. what changed, quantified: the two ports P6.2 moves move, and the three it
   does not are bit-for-bit what P6.1 produced;
4. the counterfactual: with the tone removed, a lone antagonist is inert
   again, which is what makes the whole exercise load-bearing.
"""

import numpy as np
import pytest

from neurocascade import endogenous_tone as tone
from neurocascade import monoamine_transporters as mt
from neurocascade import projection
from neurocascade import projection_scenarios as pscen
from neurocascade import projection_systems as ps
from neurocascade.receptor import Ligand, ReceptorSite


# ---------------------------------------------------------------------------
# 1. Cited basal concentrations, one per projection system
# ---------------------------------------------------------------------------


def test_every_projection_system_has_a_cited_basal_tone():
    """The first acceptance criterion, checked against the list of
    projection systems rather than against a hand-written list, so a sixth
    transmitter cannot be added without a tone."""
    assert set(tone.BASAL_TONE) == set(ps.CLEARANCE_BUILDERS)
    assert set(tone.BASAL_TONE) == set(ps.NUCLEI_BY_TRANSMITTER)
    for name, entry in tone.BASAL_TONE.items():
        assert entry.nM > 0.0, name
        assert entry.citation in tone.CITATIONS, name
        assert entry.species in ("mouse", "rat"), name
        assert entry.region and entry.method, name
        assert entry.caveats or entry.citation in ("Mathews2004",), name


def test_the_registry_agrees_with_the_constants_it_records():
    """The registry is a record OF the constants, not a second copy of
    them: a drift between the two would be two numbers for one quantity."""
    assert tone.BASAL_TONE["dopamine"].nM == mt.BASAL_DOPAMINE
    assert tone.BASAL_TONE["serotonin"].nM == mt.BASAL_SEROTONIN
    assert tone.BASAL_TONE["noradrenaline"].nM == mt.BASAL_NORADRENALINE
    assert tone.BASAL_TONE["acetylcholine"].nM == ps.ACETYLCHOLINE_BASAL
    assert tone.BASAL_TONE["histamine"].nM == ps.HISTAMINE_BASAL
    # and every clearance object a projection system is built from starts
    # its pool at the same number.
    for name, builder in ps.CLEARANCE_BUILDERS.items():
        assert builder().basal == pytest.approx(tone.BASAL_TONE[name].nM)


def test_no_unit_scale_survives_anywhere():
    """Rule 3, mechanically: the two illustrative names P6.1 left behind are
    gone from the code, in the same commit that supplies their
    measurements."""
    assert not hasattr(mt, "NORADRENALINE_BASAL_UNIT_SCALE")
    assert not hasattr(ps, "ACETYLCHOLINE_BASAL_UNIT_SCALE")
    assert mt.BASAL_NORADRENALINE == 1.59
    assert ps.ACETYLCHOLINE_BASAL == 18.4
    # and every port now reports a measured scale.
    assert set(pscen.MEASURED_BASAL) == set(ps.CLEARANCE_BUILDERS)


def test_the_two_new_constants_carry_their_citations_verbatim():
    """A number is only sourced if the quoted sentence it came from is
    where a reader will meet it."""
    assert "1.59 +/- 0.37 nM" in ps.CITATIONS["PanLai1995"]
    assert "ZERO NET FLUX" in ps.CITATIONS["PanLai1995"]
    assert "8525466" in ps.CITATIONS["PanLai1995"]
    assert "18.4 +/- 11.8 nM" in ps.CITATIONS["Vinson1997"]
    assert "9130679" in ps.CITATIONS["Vinson1997"]
    source = open(mt.__file__).read()
    assert "1.59 +/- 0.37 nM" in source
    assert "PanLai1995" in source


def test_each_new_constant_carries_the_disagreeing_cross_check():
    """P6.2's rule for a constant with a rival measurement: carry both, use
    the conservative one, and never average them into a third number that
    nobody measured."""
    na = tone.BASAL_TONE["noradrenaline"]
    ach = tone.BASAL_TONE["acetylcholine"]
    assert na.cross_check[1] == "Abercrombie1988"
    assert ach.cross_check[1] == "DeBundel2008"
    # The two disagreements point in OPPOSITE directions, and what picks the
    # used value is the METHOD (zero/no net flux needs no recovery estimate)
    # rather than the size. For noradrenaline that happens to be the smaller
    # value, which understates every claim made from it; for acetylcholine it
    # is the larger, which is why the sensitivity pass exists.
    assert na.nM < na.cross_check[0]
    assert ach.nM > ach.cross_check[0]
    assert "acetylcholine" in tone.SENSITIVITY and "noradrenaline" in tone.SENSITIVITY
    assert na.cross_check_ratio == pytest.approx(14.0 / 1.59, rel=1e-9)
    assert ach.cross_check_ratio == pytest.approx(4.0 / 18.4, rel=1e-9)
    # and neither is the mean of the two, which would be a number no
    # laboratory produced.
    for entry in (na, ach):
        assert entry.nM != pytest.approx(
            0.5 * (entry.nM + entry.cross_check[0])
        )


def test_the_acetylcholine_dialysate_bound_is_recorded_as_a_bound():
    """A dialysate concentration is a strict lower bound on the
    extracellular one, and is used only as one."""
    assert ps.ACETYLCHOLINE_BASAL_DIALYSATE_LOWER_BOUND == 0.40
    assert (
        ps.ACETYLCHOLINE_BASAL_DIALYSATE_LOWER_BOUND
        < ps.ACETYLCHOLINE_BASAL_HIPPOCAMPAL_ESTIMATE
        < ps.ACETYLCHOLINE_BASAL
    )
    assert "LOWER BOUND" in open(ps.__file__).read()


# ---------------------------------------------------------------------------
# 2. Competition in one shared pool
# ---------------------------------------------------------------------------


def _two_ligand_site(kd_tone, kd_drug):
    return ReceptorSite(
        name="test",
        ligands=[
            Ligand(name="tone", kon=0.5, koff=0.5 * kd_tone, efficacy=1.0),
            Ligand(name="drug", kon=0.3, koff=0.3 * kd_drug, efficacy=0.0),
        ],
    )


def test_the_dose_ratio_is_what_the_integrated_pool_actually_does():
    """The load-bearing check: `1 + C_tone/Kd_tone` is asserted against the
    ODE, not against the algebra it was derived from.

    A drug at concentration `D` against tone must reach exactly the
    occupancy that `D / ratio` reaches against an empty pool.
    """
    site = _two_ligand_site(kd_tone=8.0, kd_drug=3.0)
    tone_nM = 5.0
    ratio = projection.tone_dose_ratio(site, np.array([tone_nM, 0.0]), against="drug")
    assert ratio == pytest.approx(1.0 + tone_nM / 8.0)
    for dose in (0.1, 1.0, 10.0, 1000.0):
        with_tone = projection.resting_occupancy(site, np.array([tone_nM, dose]))[1]
        without = projection.resting_occupancy(site, np.array([0.0, dose / ratio]))[1]
        assert with_tone == pytest.approx(without, rel=1e-6)


def test_the_dose_ratio_does_not_depend_on_the_drug():
    """The property that makes one number per (receptor, transmitter) pair
    enough: the ratio is a function of the tone and its affinity alone."""
    ratios = []
    for kd_drug in (0.01, 1.0, 100.0, 1e5):
        site = _two_ligand_site(kd_tone=8.0, kd_drug=kd_drug)
        ratios.append(
            projection.tone_dose_ratio(site, np.array([5.0, 0.0]), against="drug")
        )
    assert ratios == pytest.approx([ratios[0]] * len(ratios), rel=1e-12)


def test_a_channel_blocker_is_left_out_of_the_competitive_pool():
    """P5.5's measurement is that a non-competitive open-channel blocker does
    NOT displace the agonist. Giving it a seat in the dose ratio would
    contradict the measurement that defines it."""
    from neurocascade.receptor import ChannelBlockingReceptorSite

    site = ChannelBlockingReceptorSite(
        name="test",
        ligands=[Ligand(name="agonist", kon=1.0, koff=10.0, efficacy=1.0)],
        channel_blockers=[Ligand(name="blocker", kon=1.0, koff=2.0, efficacy=0.0)],
    )
    assert set(projection.orthosteric_affinities(site)) == {"agonist"}
    # 1000 nM of blocker changes the competitive dose ratio not at all.
    assert projection.tone_dose_ratio(site, np.array([0.0, 1000.0])) == 1.0


def test_the_desensitizing_site_competes_at_its_apparent_affinity():
    """P5.4's desensitized state still holds the pool, so what competes is
    bound + desensitized and the right constant is `Kd / (1 + kdes/krec)`.
    Checked against the integrated site rather than asserted."""
    site = tone.TONE_TARGETS[7].site()  # alpha4beta2
    assert site.name.startswith("alpha4beta2") or "alpha4beta2" in site.name
    kd_app = projection.orthosteric_affinities(site)["acetylcholine"]
    plain_kd = None
    for ligand in site.orthosteric_ligands:
        if ligand.name == "acetylcholine":
            plain_kd = ligand.koff / ligand.kon
    assert kd_app == pytest.approx(
        plain_kd / site.apparent_affinity_factor()[0]
    )
    # the integrated pool is half-full of transmitter at the APPARENT Kd.
    names = [lig.name for lig in site.ligands]
    concentrations = np.zeros(len(names))
    concentrations[names.index("acetylcholine")] = kd_app
    state = projection.resting_occupancy(site, concentrations)
    bound, desens, _ = site.split_states(state)
    assert bound[0] + desens[0] == pytest.approx(0.5, rel=1e-4)


def test_every_tonic_occupancy_matches_the_closed_form():
    """The table `tone_report()` prints is integrated; here it is checked
    against `C/(C + Kd)` with the site's own competing affinity, which is
    the same statement the dose ratio makes seen from the other side."""
    for row, target in zip(tone.tone_table(), tone.TONE_TARGETS):
        kd = tone.transmitter_kd(target)
        expected = row["basal_nM"] / (row["basal_nM"] + kd)
        assert row["occupancy"] == pytest.approx(expected, rel=1e-4), target.name
        assert row["dose_ratio"] == pytest.approx(1.0 + row["basal_nM"] / kd)


def test_the_tone_table_covers_every_receptor_these_transmitters_bind():
    """A receptor left out of the table would be a receptor whose tone
    nobody had to think about."""
    covered = {(t.name, t.transmitter) for t in tone.TONE_TARGETS}
    assert len(covered) == len(tone.TONE_TARGETS)
    by_transmitter = {}
    for target in tone.TONE_TARGETS:
        by_transmitter.setdefault(target.transmitter, []).append(target.name)
    assert set(by_transmitter) == set(tone.BASAL_TONE)
    # every ported receptor placement appears in the table.
    ported = set()
    for key in pscen.PORT_ORDER:
        for placement in pscen.SCENARIO_PORTS[key]().placements:
            ported.add(placement.name)
    assert ported <= {t.name for t in tone.TONE_TARGETS if t.ported}
    # and the one receptor no port places is recorded as unported rather
    # than dropped.
    assert [t.name for t in tone.TONE_TARGETS if not t.ported] == ["D1"]


def test_the_transmitter_binds_the_pool_the_drug_binds():
    """'Shared occupancy pool' is a structural claim: the endogenous ligand
    and the drugs are on ONE `ReceptorSite`, sharing one free fraction."""
    for target in tone.TONE_TARGETS:
        site = target.site()
        names = [lig.name for lig in site.ligands]
        assert target.ligand in names, target.name
        assert len(names) >= 2, target.name
        # the free fraction the transmitter leaves is the one every other
        # ligand competes for: raise the tone, every drug's occupancy falls.
        drug = next(n for n in names if n != target.ligand)
        low = np.zeros(len(names))
        high = np.zeros(len(names))
        low[names.index(drug)] = high[names.index(drug)] = 100.0
        high[names.index(target.ligand)] = 1e5
        occ_low = projection.resting_occupancy(site, low)[names.index(drug)]
        occ_high = projection.resting_occupancy(site, high)[names.index(drug)]
        assert occ_high < occ_low, target.name


# ---------------------------------------------------------------------------
# 3. What changed, quantified
# ---------------------------------------------------------------------------


def test_the_three_already_measured_ports_do_not_move_at_all():
    """P6.2 must not disturb a result whose tone it did not change. Asserted
    by re-running rather than by not looking."""
    for key in tone.UNCHANGED_PORTS:
        assert key not in tone.ADDED_BY_P6_2
        change = tone.port_change(key, t_end=10.0)
        before = change["before"]["lone_antagonist"]["terminal_rate"]
        after = change["after"]["lone_antagonist"]["terminal_rate"]
        assert after == pytest.approx(before, abs=1e-12), key


def test_the_dopamine_port_still_reproduces_P5_8s_haloperidol_number():
    """P6.1's sharpest regression, re-checked after P6.2 touched the
    noradrenaline and acetylcholine constants: the dopamine port's lone
    antagonist is still P5.8's -0.200."""
    effect = pscen.lone_antagonist_effect("dopamine", t_end=30.0)
    assert effect["terminal_rate"] == pytest.approx(-0.20005, abs=5e-5)


def test_noradrenaline_tone_rises_by_the_measured_factor():
    """1.59 nM against P6.1's 1.0 nM unit scale. At alpha1 and beta, which
    sit four and three orders of magnitude below their Kd, the response is
    linear and the occupancy rises by exactly 1.59; at the alpha2
    autoreceptor, which is only 17-fold below its Kd, saturation has begun
    and the rise is 1.56. That difference is the whole reason tonic
    occupancy is computed rather than scaled."""
    old = {r["receptor"]: r for r in tone.tone_table(basal={"noradrenaline": 1.0})}
    new = {r["receptor"]: r for r in tone.tone_table()}
    for name, expected in (("alpha1", 1.59), ("beta", 1.59)):
        ratio = new[name]["occupancy"] / old[name]["occupancy"]
        assert ratio == pytest.approx(expected, rel=2e-3), name
    ratio_alpha2 = new["alpha2"]["occupancy"] / old["alpha2"]["occupancy"]
    assert 1.55 < ratio_alpha2 < 1.57
    assert ratio_alpha2 < 1.59
    assert new["alpha2"]["occupancy"] == pytest.approx(0.0556, abs=5e-4)


def test_acetylcholine_tone_rises_by_the_measured_factor():
    """18.4 nM against P6.1's 1.0 nM. Linear at alpha7, M1 and M2; sublinear
    at alpha4beta2, whose apparent affinity is tightened 24.5-fold by P5.4's
    desensitized state and which is therefore no longer in the linear
    regime."""
    old = {r["receptor"]: r for r in tone.tone_table(basal={"acetylcholine": 1.0})}
    new = {r["receptor"]: r for r in tone.tone_table()}
    for name in ("alpha7", "M1", "M2"):
        ratio = new[name]["occupancy"] / old[name]["occupancy"]
        assert ratio == pytest.approx(18.4, rel=5e-3), name
    ratio_a4b2 = new["alpha4beta2"]["occupancy"] / old["alpha4beta2"]["occupancy"]
    assert 14.4 < ratio_a4b2 < 14.7
    assert ratio_a4b2 < 18.4
    assert new["alpha4beta2"]["occupancy"] == pytest.approx(0.2197, abs=1e-3)


def test_the_two_moved_ports_move_in_the_reported_direction():
    """The change P6.2 is asked to report: both lone antagonists get
    stronger, because both pools are fuller than they were."""
    for key in tone.CHANGED_PORTS:
        change = tone.port_change(key, t_end=30.0)
        before = change["before"]["lone_antagonist"]["terminal_rate"]
        after = change["after"]["lone_antagonist"]["terminal_rate"]
        assert abs(after) > abs(before), key
        assert np.sign(after) == np.sign(before), key
    na = tone.port_change("noradrenaline", t_end=30.0)
    ach = tone.port_change("acetylcholine", t_end=30.0)
    assert (
        na["after"]["lone_antagonist"]["terminal_rate"]
        / na["before"]["lone_antagonist"]["terminal_rate"]
    ) == pytest.approx(2.4, rel=0.15)
    assert (
        ach["after"]["lone_antagonist"]["terminal_rate"]
        / ach["before"]["lone_antagonist"]["terminal_rate"]
    ) == pytest.approx(15.5, rel=0.15)


def test_the_field_fold_is_basal_independent_at_a_FIXED_source_rate():
    """The invariant P5.8 and P6.1 both relied on, restated precisely.

    A first-order or Michaelis-Menten pool's fold response to a given
    firing-rate change is independent of its basal level. That is a
    statement about the pool AT A HELD SOURCE RATE, and it survives P6.2
    exactly: quadrupling or quartering the tone moves no fold at all.
    """
    for key, old_basal in tone.ADDED_BY_P6_2.items():
        folds = []
        for basal in (old_basal, tone.BASAL_TONE[key].nM, 100.0 * old_basal):
            system = pscen.port(key, basal=basal).projection
            rates = np.full(len(system.sources), 0.25)
            folds.append(
                system.steady_state("PFC", rates)
                / system.basal_concentration("PFC")
            )
        for other in folds[1:]:
            assert other == pytest.approx(folds[0], rel=1e-9), key


def test_the_closed_loop_field_fold_DOES_move_and_the_reason_is_the_loop():
    """The counterpart, and the thing it would have been easy to get wrong.

    In a CLOSED loop the fold does move, because the firing-rate change is
    no longer fixed: raising the tone raises the autoreceptor's tonic
    occupancy, which changes how much the same drug moves the source
    nucleus, which changes the field. So the invariance above and this
    movement are both true, of two different quantities, and confusing them
    would let a real change be reported as no change.
    """
    moved = {}
    for key, old_basal in tone.ADDED_BY_P6_2.items():
        new = pscen.closed_loop_effect(key, t_end=30.0)
        old = pscen.closed_loop_effect(key, t_end=30.0, basal=old_basal)
        moved[key] = (new["terminal_field_fold"], old["terminal_field_fold"])
    # noradrenaline: clonidine's grip on the LC changes with alpha2 tone, so
    # the halving of the cortical field is 6.5% shallower at the measured
    # tone than at P6.1's unit scale.
    new_na, old_na = moved["noradrenaline"]
    assert new_na != pytest.approx(old_na, rel=1e-3)
    assert abs(new_na) < abs(old_na)
    assert new_na / old_na == pytest.approx(0.935, rel=0.02)


# ---------------------------------------------------------------------------
# 4. The counterfactual: an empty pool
# ---------------------------------------------------------------------------


def test_with_the_tone_removed_every_lone_antagonist_is_inert_again():
    """The sentence every module from P5.1 to P5.7 had to write, reproduced
    on demand. Without it the claim 'tone is what makes an antagonist do
    anything' would be an assertion rather than a measurement."""
    for key in pscen.PORT_ORDER:
        effect = tone.empty_pool_change(key, t_end=30.0)
        assert abs(effect["empty_pool"]) < 1e-6, key
        assert abs(effect["with_tone"]) > abs(effect["empty_pool"]), key


def test_the_size_of_the_lone_antagonist_effect_tracks_the_tonic_occupancy():
    """What an antagonist can do is bounded by what tone it can remove, so
    the ordering of the ports' effects follows the ordering of the tonic
    occupancy at the receptor each antagonist binds.

    P6.3 changed WHICH concentration that occupancy has to be computed at,
    and this test is where the change shows. `tone_table()` reports the tone
    at each transmitter's MEASURED basal level, which is the concentration
    in its REFERENCE region; before P6.3's derived density map that was the
    concentration everywhere, so the two questions had one answer. They no
    longer do: the cortical H1 receptor sees 0.016 nM of histamine rather
    than the hypothalamus's measured 7.8 nM, and the cortical M1 receptor
    sees 10.1 nM of acetylcholine rather than the striatum's 18.4 nM. So the
    ordering of the EFFECTS follows the occupancy at each receptor's OWN
    REGION, and the test asserts that instead. It is a stronger statement
    than the one it replaces, because the two orderings now differ: at the
    reference concentrations H1 outranks M1, and in the regions the ports
    actually place them M1 outranks H1, and it is the second that predicts
    the effects."""
    reference = {row["receptor"]: row["occupancy"] for row in tone.tone_table()}
    # Unchanged, and no longer the right question for a ported scenario.
    assert reference["D2"] > reference["H1"] > reference["M1"]

    regional = {}
    for key, receptor, region in (
        ("dopamine", "D2", "CP"),
        ("histamine", "H1", "PFC"),
        ("acetylcholine", "M1", "PFC"),
    ):
        scenario = pscen.port(key)
        level = scenario.projection.basal_concentration(region)
        target = next(t for t in tone.TONE_TARGETS if t.name == receptor)
        regional[receptor] = tone.tonic_occupancy(target, basal=level)
    assert regional["D2"] > regional["M1"] > regional["H1"]

    effects = {
        key: abs(pscen.lone_antagonist_effect(key, t_end=30.0)["terminal_rate"])
        for key in ("dopamine", "histamine", "acetylcholine")
    }
    assert effects["dopamine"] > effects["acetylcholine"] > effects["histamine"]


# ---------------------------------------------------------------------------
# 5. Reports
# ---------------------------------------------------------------------------


def test_the_reports_render_and_name_their_sources():
    text = tone.basal_report()
    for citation in ("Bosse2011", "Mathews2004", "PanLai1995", "Vinson1997", "Itoh1991"):
        assert citation in text
    table = tone.tone_report()
    for target in tone.TONE_TARGETS:
        assert target.name in table
    assert "dose ratio" in table
    sensitivity = tone.sensitivity_report()
    assert "Abercrombie1988" in sensitivity and "DeBundel2008" in sensitivity


def test_the_module_records_the_acetylcholine_region_mismatch():
    """The weakest thing about P6.2's constants has to be visible where a
    reader will meet it, not only in docs/architecture.md."""
    doc = tone.__doc__
    assert "STRIATUM" in doc and "INTERNEURONS" in doc
    assert "interneurons" in " ".join(
        tone.BASAL_TONE["acetylcholine"].caveats
    )
