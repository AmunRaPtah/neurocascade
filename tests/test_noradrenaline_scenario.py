"""Tests for the P5.3 noradrenaline alpha1 / alpha2 / beta receptor systems.

`noradrenaline.py`'s module docstring carries the full source list and the
derivation of every constant; each test below restates the published figure
or the stated design rule it checks against, so a silent edit to the module
cannot pass.

Four things are being checked, and they are different in kind:

  * the BINDING layer against published numbers ([Proudman2020],
    [Proudman2021], [Proudman2022], [Proudman2022b], [Baker2005],
    [Baker2010], [Sykes2022], [Tanaka2004], [Quinton2010], [Gerhardt1990],
    [UPrichard1978], [Salama1982]);
  * the CIRCUIT layer against the three stated illustrative-parameter rules
    (rest at maximum sensitivity, P3.4's contraction certificate,
    no saturation) -- not against literature, because it has none;
  * the PHARMACOLOGY against the qualitative predictions of adrenoceptor
    physiology ([McCormick1991], [Madison1986], [Cedarbaum1977]);
  * the two documented AFFINITY-STATE choices, with every qualitative
    result asserted under both. Where the whole-cell noradrenaline set
    cannot be integrated -- which is itself a measured fact this file
    asserts arithmetically -- it is checked at the algebraic fixed point
    instead.
"""

import numpy as np
import pytest

from neurocascade import dopamine, noradrenaline, serotonin, stability
from neurocascade.engine import ReceptorSystem, Simulation
from neurocascade.pk import OneCompartmentPK
from neurocascade.receptor import ReceptorSite

T_SPAN = (0.0, 10.0)
T_EVAL = np.linspace(0.0, 10.0, 200)

#: nM. The noradrenaline dose the dynamic tests use, and the reason it is
#: not larger. At 3000 nM the equilibrium occupancies are 0.16 at alpha1A
#: (Kd 15488 nM) and 0.62 at beta1 (Kd 1820 nM), so both cortical
#: receptors are engaged and beta1 more than alpha1, which is what these
#: tests need. Raising it costs integrator time superlinearly, because the
#: alpha2 site's shared kon (0.48 nM^-1 min^-1) makes the pseudo-first-order
#: association rate 0.48 * C: at 3000 nM that is already 1440 min^-1 and
#: RK45's step is bounded by it. `test_the_whole_cell_alpha2_noradrenaline_
#: set_is_not_integrable` is the extreme case of the same arithmetic.
NORADRENALINE_DOSE_nM = 3000.0

#: nM. Doses of the three dosed antagonists/agonists. Each is far above its
#: own Kd (prazosin 0.90 nM, propranolol 0.71 nM, clonidine 5.9 nM in the
#: default affinity state), so each takes essentially all of its pool.
PRAZOSIN_DOSE_nM = 100.0
PROPRANOLOL_DOSE_nM = 100.0
CLONIDINE_DOSE_nM = 1000.0
YOHIMBINE_DOSE_nM = 1000.0


def by_name(ligands):
    return {ligand.name: ligand for ligand in ligands}


def kd(ligand):
    return ligand.koff / ligand.kon


def equilibrium_occupancy(ligand, concentration):
    """The single-ligand equilibrium `C / (C + Kd)`, verified to BE the
    fixed point of `ReceptorSite.occupancy_rhs` rather than assumed.

    P5.2's helper, reused for the same reason and for one more: it is the
    only way to state anything about noradrenaline at the whole-cell alpha2
    affinity, whose 129194 min^-1 koff cannot be integrated (see the test
    that measures that).
    """
    occupancy = concentration / (concentration + kd(ligand))
    site = ReceptorSite(name="site", ligands=[ligand])
    residual = site.occupancy_rhs([occupancy], [concentration])[0]
    assert abs(residual) < 1e-12
    return occupancy


def competitive_equilibrium(ligands, concentrations):
    """Fractional occupancy per ligand at the competitive fixed point.

    For a shared pool, `occ_i = (C_i/Kd_i) / (1 + sum_j C_j/Kd_j)`. As in
    `equilibrium_occupancy`, the result is verified to be the fixed point of
    the site's own ODE rather than asserted, so this helper cannot drift
    away from `ReceptorSite`.
    """
    ratios = [c / kd(lig) for lig, c in zip(ligands, concentrations)]
    denominator = 1.0 + sum(ratios)
    occupancies = [r / denominator for r in ratios]
    site = ReceptorSite(name="site", ligands=list(ligands))
    residual = np.max(np.abs(site.occupancy_rhs(occupancies, concentrations)))
    assert residual < 1e-9
    return occupancies


def build_simulation(affinity_state="agonist-state", propranolol_affinity="kinetic"):
    network = noradrenaline.locus_coeruleus_cortical_network()
    systems = noradrenaline.noradrenergic_receptor_systems(
        affinity_state=affinity_state, propranolol_affinity=propranolol_affinity
    )
    return Simulation(network=network, receptor_systems=systems)


def traces(sim, **doses):
    """Run from the network's rest state and return (result, per-node rates).

    Starting at `rest_rates()` rather than at zero rates is what makes the
    drug-free trace exactly flat, so every deviation below is the drug and
    not the start-up transient (see `noradrenaline.rest_rates`).
    """
    result = sim.run(
        dose_concentrations=sim.dose_vector(**doses),
        t_span=T_SPAN,
        t_eval=T_EVAL,
        rates0=noradrenaline.rest_rates(sim.network),
    )
    idx = sim.network.index
    return result, {name: result["rates"][i] for name, i in idx.items()}


@pytest.fixture(scope="module")
def sim():
    """One `Simulation` for the dynamic tests; building it is cheap but
    every `run` is not, so the tests below share the object and not the
    runs."""
    return build_simulation()


# ---------------------------------------------------------------------------
# The binding layer: does the module reproduce the published numbers?
# ---------------------------------------------------------------------------


def test_every_affinity_is_the_published_log_kd():
    # Every affinity in this module is a Baker-laboratory log KD (log10 of
    # molar Kd) turned into nM. If the conversion is wrong every result is
    # wrong, so it is checked against the printed values one at a time:
    #   alpha1A  noradrenaline  -4.81  [Proudman2021]   prazosin -9.07 [Proudman2020]
    #   alpha2A  noradrenaline  -3.57  [Proudman2022]   clonidine -6.72 [Proudman2022]
    #                                                   yohimbine -8.48 [Proudman2022b]
    #   beta1    noradrenaline  -5.74  [Baker2010]      propranolol -8.16 [Baker2005]
    published_nM = {
        noradrenaline.ALPHA1_KD_NORADRENALINE: 15488.17,
        noradrenaline.ALPHA1_KD_PRAZOSIN_BINDING: 0.8511,
        noradrenaline.ALPHA2_KD_NORADRENALINE_WHOLE_CELL: 269153.5,
        noradrenaline.ALPHA2_KD_CLONIDINE_WHOLE_CELL: 190.546,
        noradrenaline.ALPHA2_KD_YOHIMBINE: 3.31131,
        noradrenaline.BETA_KD_NORADRENALINE: 1819.70,
        noradrenaline.BETA_KD_PROPRANOLOL_WHOLE_CELL: 6.91831,
    }
    for value, expected in published_nM.items():
        assert abs(value - expected) < 1e-4 * expected

    # ...and each ligand's own Kd is where its occupancy is a half, by
    # definition, which is what ties the constants above to the ODE.
    every_ligand = (
        noradrenaline.ligands_alpha1()
        + noradrenaline.ligands_alpha2()
        + noradrenaline.ligands_beta()
    )
    for ligand in every_ligand:
        assert abs(equilibrium_occupancy(ligand, kd(ligand)) - 0.5) < 1e-12


def test_the_recorded_subtype_panels_are_the_published_numbers():
    # The constants this module keeps but does not instantiate. They are the
    # error bars on the claims it does make, so a silent edit to any of them
    # has to fail something.
    #
    # Yohimbine, measured twice at alpha2A in ONE paper against two
    # radioligands [Proudman2022b]: log KD -8.48 ([3H]rauwolscine) and -8.58
    # ([3H]RX821002). A 1.26-fold spread, the smallest honest error bar in
    # this module -- against 10000-fold for noradrenaline at the same site.
    rauwolscine_kd = noradrenaline.ALPHA2_KD_YOHIMBINE
    rx821002_kd = 10.0 ** (9.0 + noradrenaline.ALPHA2A_LOG_KD_YOHIMBINE_RX821002)
    assert abs(rauwolscine_kd / rx821002_kd - 1.259) < 0.005
    state_spread = (
        noradrenaline.ALPHA2_KD_NORADRENALINE_WHOLE_CELL
        / noradrenaline.ALPHA2_KD_NORADRENALINE_AGONIST_STATE
    )
    assert state_spread > 1000.0 * (rauwolscine_kd / rx821002_kd)

    # Clonidine's two 3H-clonidine sites in rat cortex [Salama1982]: ~1.0 nM
    # super-high and ~5.9 nM high. The "high" one is used, because
    # [Salama1982] shows the super-high component is the GTP-sensitive one.
    assert noradrenaline.ALPHA2_KD_CLONIDINE_SUPERHIGH_STATE == 1.0
    assert (
        noradrenaline.ALPHA2_KD_CLONIDINE_SUPERHIGH_STATE
        < noradrenaline.ALPHA2_KD_CLONIDINE_AGONIST_STATE
    )

    # Noradrenaline across the beta subtypes [Baker2010]: -5.74 / -5.41 /
    # -5.53, i.e. barely selective at all (2.1-fold), which is why "beta1"
    # rather than "beta2" is a labelling choice here and not a claim.
    beta_kds = [
        10.0 ** (9.0 + log_kd)
        for log_kd in (
            noradrenaline.BETA1_LOG_KD_NORADRENALINE,
            noradrenaline.BETA2_LOG_KD_NORADRENALINE,
            noradrenaline.BETA3_LOG_KD_NORADRENALINE,
        )
    ]
    assert max(beta_kds) / min(beta_kds) < 2.2

    # Propranolol across the beta subtypes [Baker2005]: -8.16 / -9.08 /
    # -6.93. beta2 is TIGHTER than beta1 by 8.3-fold, so propranolol is
    # beta2-PREFERRING rather than beta1-selective, which is the source of
    # its bronchospasm liability and a fact the single-site model here
    # cannot express.
    beta1_kd = 10.0 ** (9.0 + noradrenaline.BETA1_LOG_KD_PROPRANOLOL)
    beta2_kd = 10.0 ** (9.0 + noradrenaline.BETA2_LOG_KD_PROPRANOLOL)
    beta3_kd = 10.0 ** (9.0 + noradrenaline.BETA3_LOG_KD_PROPRANOLOL)
    assert beta2_kd < beta1_kd < beta3_kd
    assert abs(beta1_kd / beta2_kd - 8.32) < 0.05

    # And at every beta subtype propranolol is far tighter than
    # noradrenaline, which is what makes the reversal result robust to the
    # subtype label.
    for antagonist_kd, agonist_kd in zip(
        (beta1_kd, beta2_kd, beta3_kd), beta_kds
    ):
        assert agonist_kd / antagonist_kd > 25.0


def test_rate_constant_unit_conversions_are_the_published_numbers():
    # The three measured rate constants, from their papers' own units into
    # this codebase's nM^-1 min^-1 and min^-1.
    #   propranolol at beta1     1.28e9 M^-1 min^-1, 0.91 min^-1  [Sykes2022]
    #   isoprenaline at beta1    2.51e7 M^-1 min^-1               [Sykes2022]
    #   p-iodoclonidine at alpha2  8.0e6 M^-1 s^-1                [Gerhardt1990]
    #   [3H]prazosin at alpha1   already in nM^-1 min^-1          [Tanaka2004]
    assert abs(noradrenaline.BETA_KON_PROPRANOLOL - 1.28e9 * 1e-9) < 1e-12
    assert noradrenaline.BETA_KOFF_PROPRANOLOL == 0.91
    assert abs(noradrenaline.BETA_KON_ISOPRENALINE - 2.51e7 * 1e-9) < 1e-12
    assert abs(noradrenaline.ALPHA2_KON - 8.0e6 * 60 * 1e-9) < 1e-12
    assert noradrenaline.ALPHA1_KON_PRAZOSIN == 0.048

    # Derivation 3 is exactly kon * Kd at each site, so the derived koffs
    # have to be reproducible from the two published inputs.
    assert (
        abs(
            noradrenaline.ALPHA1_KOFF_NORADRENALINE
            - 0.048 * noradrenaline.ALPHA1_KD_NORADRENALINE
        )
        < 1e-12
    )
    assert (
        abs(
            noradrenaline.BETA_KOFF_NORADRENALINE
            - 0.0251 * noradrenaline.BETA_KD_NORADRENALINE
        )
        < 1e-12
    )


def test_propranolol_kinetic_pair_reproduces_its_own_papers_equilibrium_pkd():
    # [Sykes2022] reports koff, kon AND pKd for propranolol at human beta1
    # as three independent entries. koff/kon = 0.911/1.28 = 0.711 nM against
    # the reported pKd of 9.13 (0.741 nM): they agree to 4%, which is the
    # units check that the kinetic pair is being read correctly.
    propranolol = by_name(noradrenaline.ligands_beta())["propranolol"]
    kinetic_kd = kd(propranolol)
    reported_kd = 10.0 ** (9.0 - 9.13)
    assert abs(kinetic_kd - 0.710938) < 1e-6
    assert abs(kinetic_kd - reported_kd) < 0.05 * reported_kd

    # Residence time 1/koff = 1.1 min, which is what [Sykes2022] prints.
    assert abs(1.0 / propranolol.koff - 1.10) < 0.01

    # This makes propranolol the THIRD kind of antagonist in this codebase:
    # naloxone (koff 2.375) is fast-off and reverses only in molar excess,
    # WAY-100635 (0.0078) and SCH-23390 (0.00389) are near-irreversible.
    # Propranolol sits between them, two orders faster off than either slow
    # antagonist, which is why its reversal below is prompt.
    way = by_name(serotonin.ligands_5ht1a())["WAY-100635"]
    sch = by_name(dopamine.d1_dopamine_ligands())["SCH-23390"]
    assert propranolol.koff > 100 * way.koff
    assert propranolol.koff > 100 * sch.koff


def test_prazosin_is_the_one_constant_three_measurements_agree_on():
    # kon  0.048 nM^-1 min^-1  rat tail artery segments, 4 C   [Tanaka2004]
    # koff 0.043 min^-1        rat brain synaptosomes, RT      [Quinton2010]
    # Kd   0.851 nM            human alpha1A CHO cells, 37 C   [Proudman2020]
    # Three preparations, two species, three temperatures, 5.3% apart. This
    # is derivation 1 assembled from two papers and checked against a third.
    prazosin = by_name(noradrenaline.ligands_alpha1())["prazosin"]
    assert prazosin.kon == noradrenaline.ALPHA1_KON_PRAZOSIN
    assert prazosin.koff == noradrenaline.ALPHA1_KOFF_PRAZOSIN
    kinetic_kd = kd(prazosin)
    binding_kd = noradrenaline.ALPHA1_KD_PRAZOSIN_BINDING
    assert abs(kinetic_kd - 0.8958333) < 1e-6
    assert abs(kinetic_kd - binding_kd) < 0.06 * binding_kd

    # The koff used is [Quinton2010]'s, not [Tanaka2004]'s, because
    # [Tanaka2004]'s three numbers are mutually inconsistent: it reports
    # Kon 0.048, Koff 0.078 and a kinetic pKD of 9.3, but 0.078/0.048 is
    # 1.63 nM (pKD 8.79), a 3.2-fold disagreement. The value adopted is
    # bracketed by that paper's two readings, which is the check that
    # preferring the brain-tissue measurement did not smuggle in an
    # outlier. The bracket's own arithmetic is asserted so the claim
    # cannot rot.
    tanaka_koff = 0.078
    tanaka_koff_implied_by_its_pkd = 0.048 * 10.0 ** (9.0 - 9.3)
    assert abs(tanaka_koff / (0.048 * 10.0 ** (9.0 - 9.3)) - 3.25) < 0.05
    assert tanaka_koff_implied_by_its_pkd < prazosin.koff < tanaka_koff


def test_the_sykes_panel_is_what_licenses_derivation_threes_refinement():
    # Derivation 3 assumes ligands at one site associate at comparable
    # rates. P5.1 and P5.2 could not check that; [Sykes2022]'s 18-ligand
    # beta1 panel can, and it says the assumption fails across chemotypes
    # and holds within one:
    #   whole panel      sotalol 8.47e6 ... carvedilol 1.97e9 M^-1 min^-1
    #   catecholamine-like agonists 1.68e7 ... 2.63e7
    panel_spread = (
        noradrenaline.BETA_KON_CARVEDILOL / noradrenaline.BETA_KON_SOTALOL
    )
    assert panel_spread > 200.0

    catecholamine_kons = [
        noradrenaline.BETA_KON_ISOPRENALINE,
        noradrenaline.BETA_KON_SALBUTAMOL,
        noradrenaline.BETA_KON_FORMOTEROL,
        noradrenaline.BETA_KON_SALMETEROL,
    ]
    chemotype_spread = max(catecholamine_kons) / min(catecholamine_kons)
    assert chemotype_spread < 1.7

    # Which is why noradrenaline at beta1 shares ISOPRENALINE's kon and not
    # propranolol's, 51-fold away in the same table.
    beta = by_name(noradrenaline.ligands_beta())
    assert beta["noradrenaline"].kon == noradrenaline.BETA_KON_ISOPRENALINE
    assert (
        beta["propranolol"].kon / beta["noradrenaline"].kon > 50.0
    )

    # And why the affinity, not the rate, is what the derived koffs are
    # trusted for: koff = kon * Kd reproduces the measured Kd EXACTLY for
    # any shared kon, so a wrong shared kon costs a timescale and not an
    # equilibrium. Asserted by rebuilding the same ligand with a kon 100x
    # away and checking the Kd is unchanged.
    noisy = type(beta["noradrenaline"])(
        name="noradrenaline",
        kon=100.0 * noradrenaline.BETA_KON_ISOPRENALINE,
        koff=100.0 * noradrenaline.BETA_KOFF_NORADRENALINE,
        efficacy=1.0,
    )
    assert abs(kd(noisy) - kd(beta["noradrenaline"])) < 1e-9


def test_efficacies_are_the_published_ones_and_which_zeros_are_measured():
    alpha1 = by_name(noradrenaline.ligands_alpha1())
    alpha2 = by_name(noradrenaline.ligands_alpha2())
    beta = by_name(noradrenaline.ligands_beta())

    # Noradrenaline is the reference of every panel used here, so 1.0 is a
    # definition at alpha1A and alpha2A -- but at beta1 it is a
    # MEASUREMENT: 102.4 +/- 3.6% of isoprenaline's maximum in cAMP
    # accumulation, n=10 [Baker2010]. The 1.024 is recorded so that
    # "noradrenaline is a full agonist at beta1" is sourced rather than
    # assumed, and 1.0 rather than 1.024 is used because it is the
    # reference of this module's own scale.
    for ligands in (alpha1, alpha2, beta):
        assert ligands["noradrenaline"].efficacy == 1.0
    assert noradrenaline.BETA_EMAX_NORADRENALINE_VS_ISOPRENALINE == 1.024
    assert abs(noradrenaline.BETA_EMAX_NORADRENALINE_VS_ISOPRENALINE - 1.0) < 0.05

    # Clonidine: MEASURED intrinsic activity 0.32 relative to noradrenaline
    # in one [35S]GTPgammaS panel at cloned human alpha2a [Jasper1998]. The
    # lowest fractional efficacy in the codebase (after 8-OH-DPAT 0.85, DOI
    # 0.813, dihydrexidine 0.808), and a weak partial agonist under every
    # one of the three published readings (0.32, 0.027, ~0).
    assert alpha2["clonidine"].efficacy == 0.32
    assert 0.0 < alpha2["clonidine"].efficacy < alpha2["noradrenaline"].efficacy

    # Yohimbine's zero is MEASURED: [Jasper1998] reports that antagonists
    # did not increase [35S]GTPgammaS binding at any alpha2 subtype, and
    # [Millan2000] finds antagonist actions at h-alpha2A. Same standing as
    # WAY-100635's zero in P5.1.
    assert alpha2["yohimbine"].efficacy == 0.0

    # Prazosin's and propranolol's zeros are the flagged MODELLING
    # SIMPLIFICATION instead: both are reported as inverse agonists at
    # constitutively active receptors ([Baker2003] for propranolol), and
    # `ReceptorSite.activation` has no constitutive term for a negative
    # efficacy to act against.
    assert alpha1["prazosin"].efficacy == 0.0
    assert beta["propranolol"].efficacy == 0.0


def test_the_affinity_ordering_is_a_published_result_and_the_model_has_it():
    # The one prediction in this module free of illustrative parameters, and
    # the reason all three sites' affinities were taken from one
    # laboratory's whole-cell binding programme: noradrenaline's affinity
    # for beta1 is HIGHER than for either alpha, which is not the textbook
    # ordering.
    #   beta1   1820 nM   [Baker2010]
    #   alpha1A 15488 nM  [Proudman2021]
    #   alpha2A 269153 nM [Proudman2022]
    beta_kd = noradrenaline.BETA_KD_NORADRENALINE
    alpha1_kd = noradrenaline.ALPHA1_KD_NORADRENALINE
    alpha2_kd = noradrenaline.ALPHA2_KD_NORADRENALINE_WHOLE_CELL
    assert beta_kd < alpha1_kd < alpha2_kd
    assert abs(alpha1_kd / beta_kd - 8.5) < 0.1
    assert abs(alpha2_kd / beta_kd - 147.9) < 0.5

    # It survives the alpha1 subtype uncertainty, which is why the claim is
    # made in this form: beta1 is tighter than all THREE alpha1 subtypes,
    # by 8.5-fold (alpha1A), 89-fold (alpha1B) and 1.8-fold (alpha1D). The
    # margin is subtype-dependent; the ordering is not.
    margins = []
    for log_kd in (
        noradrenaline.ALPHA1A_LOG_KD_NORADRENALINE,
        noradrenaline.ALPHA1B_LOG_KD_NORADRENALINE,
        noradrenaline.ALPHA1D_LOG_KD_NORADRENALINE,
    ):
        subtype_kd = 10.0 ** (9.0 + log_kd)
        assert subtype_kd > beta_kd
        margins.append(subtype_kd / beta_kd)
    assert abs(min(margins) - 1.8) < 0.1
    assert abs(max(margins) - 89.1) < 0.5

    # Prazosin, by contrast, is essentially non-selective across the three
    # (0.85 / 1.8 / 0.85 nM), which is what makes every prazosin result
    # below subtype-independent.
    prazosin_kds = [
        10.0 ** (9.0 + log_kd)
        for log_kd in (
            noradrenaline.ALPHA1A_LOG_KD_PRAZOSIN,
            noradrenaline.ALPHA1B_LOG_KD_PRAZOSIN,
            noradrenaline.ALPHA1D_LOG_KD_PRAZOSIN,
        )
    ]
    assert max(prazosin_kds) / min(prazosin_kds) < 2.2


def test_the_potency_ordering_is_different_and_the_model_does_not_have_it():
    # The honest counterpart of the test above, and the same limitation P5.2
    # flagged for dopamine's 530-fold-affinity / 31-fold-potency gap. From
    # the SAME papers' functional columns:
    #   alpha1A  log EC50 -8.61  (Ca2+ mobilization) [Proudman2021]  2.45 nM
    #   beta1    log EC50 -7.94  (cAMP)              [Baker2010]    11.5 nM
    #   alpha2A  log IC50 -6.60  (Gi, CRE-SPAP)      [Proudman2022] 251 nM
    # Affinity says beta1 first; potency says alpha1A first. The difference
    # is receptor reserve, worth 3.8 log units at alpha1A, and this model's
    # activation is linear in occupancy with no amplification term, so it
    # reproduces the affinity ordering and not this one.
    alpha1_ec50 = 10.0 ** (9.0 - 8.61)
    beta_ec50 = 10.0 ** (9.0 - 7.94)
    alpha2_ic50 = 10.0 ** (9.0 - 6.60)
    assert alpha1_ec50 < beta_ec50 < alpha2_ic50

    # The intrinsic efficacy ratios the same papers report are the size of
    # the gap: alpha1A 3.80, alpha2A 3.03, beta1 2.20 log units.
    for site_kd, potency, log_ratio in (
        (noradrenaline.ALPHA1_KD_NORADRENALINE, alpha1_ec50, 3.80),
        (noradrenaline.BETA_KD_NORADRENALINE, beta_ec50, 2.20),
        (noradrenaline.ALPHA2_KD_NORADRENALINE_WHOLE_CELL, alpha2_ic50, 3.03),
    ):
        assert abs(np.log10(site_kd / potency) - log_ratio) < 0.02

    # The model's occupancy at a concentration between the two orderings'
    # crossover is the concrete consequence: at 3000 nM noradrenaline the
    # beta1 pool is more occupied than the alpha1A pool, even though
    # alpha1A is the more POTENT site in the assay that measured both.
    alpha1_occ = equilibrium_occupancy(
        by_name(noradrenaline.ligands_alpha1())["noradrenaline"],
        NORADRENALINE_DOSE_nM,
    )
    beta_occ = equilibrium_occupancy(
        by_name(noradrenaline.ligands_beta())["noradrenaline"],
        NORADRENALINE_DOSE_nM,
    )
    assert beta_occ > alpha1_occ
    assert abs(alpha1_occ - 0.1622) < 0.001
    assert abs(beta_occ - 0.6224) < 0.001


def test_the_alpha2_affinity_state_choice_flips_which_receptor_comes_first():
    # The largest real uncertainty in this task, stated as the two published
    # numbers it is:
    #   whole-cell competition vs [3H]rauwolscine, human alpha2A: 269153 nM
    #     [Proudman2022]
    #   saturation binding of (-)-3H-noradrenaline itself, rat cortex: 27 nM
    #     [UPrichard1978]
    # 10000-fold. The IUPHAR/BPS Guide to PHARMACOLOGY records the published
    # range for noradrenaline at HUMAN alpha2A as pKi 3.6-7.4, and the two
    # ends of that range are worth checking against rather than quoting:
    #   * its LOOSE end IS the whole-cell value, rounded to one decimal
    #     place (-3.57 -> 3.6), i.e. [Proudman2022] is where the Guide's
    #     range bottoms out;
    #   * its TIGHT end (pKi 7.4, Kd 39.8 nM) is 1.5-fold LOOSER than the
    #     rat-cortex agonist-state value, so 27 nM lies just outside it.
    # That is the expected direction, because the Guide's range is
    # human-receptor values and [UPrichard1978] is rat cortex. Asserted in
    # this form rather than as a brackets-both claim, which would have been
    # convenient and false.
    whole_cell = noradrenaline.ALPHA2_KD_NORADRENALINE_WHOLE_CELL
    agonist_state = noradrenaline.ALPHA2_KD_NORADRENALINE_AGONIST_STATE
    assert abs(whole_cell / agonist_state - 9968.6) < 1.0
    assert round(np.log10(whole_cell * 1e-9), 1) == -3.6
    guide_tight_end = 10.0 ** (9.0 - 7.4)
    assert agonist_state < guide_tight_end
    assert abs(guide_tight_end / agonist_state - 1.47) < 0.05

    # And the consequence: the state choice, not the literature, decides
    # which adrenoceptor noradrenaline engages first in this model.
    beta_kd = noradrenaline.BETA_KD_NORADRENALINE
    assert whole_cell > beta_kd  # whole-cell: beta1 first
    assert agonist_state < beta_kd  # agonist-state: alpha2 first
    assert abs(beta_kd / agonist_state - 67.4) < 0.5

    # Both sets are buildable and each ligand's Kd comes back out exactly.
    expected = {
        "agonist-state": (agonist_state, noradrenaline.ALPHA2_KD_CLONIDINE_AGONIST_STATE),
        "whole-cell": (whole_cell, noradrenaline.ALPHA2_KD_CLONIDINE_WHOLE_CELL),
    }
    for state, (na_kd, clonidine_kd) in expected.items():
        ligands = by_name(noradrenaline.ligands_alpha2(affinity_state=state))
        assert abs(kd(ligands["noradrenaline"]) - na_kd) < 1e-9 * na_kd
        assert abs(kd(ligands["clonidine"]) - clonidine_kd) < 1e-9 * clonidine_kd
        # Yohimbine is the same in both, and clonidine is tighter than
        # noradrenaline under both, which is what makes the qualitative
        # results below state-independent.
        assert abs(kd(ligands["yohimbine"]) - noradrenaline.ALPHA2_KD_YOHIMBINE) < 1e-12
        assert kd(ligands["clonidine"]) < kd(ligands["noradrenaline"])

    with pytest.raises(ValueError):
        noradrenaline.ligands_alpha2(affinity_state="whatever-fits")


def test_the_whole_cell_alpha2_noradrenaline_set_is_not_integrable():
    # Recorded as an assertion rather than as a paragraph, because it is the
    # reason the whole-cell alpha2 results below are checked at the
    # algebraic fixed point instead of by integration.
    #
    # koff = ALPHA2_KON * 269153 = 129194 min^-1, a relaxation time of
    # 0.3 ms. `Simulation.run` integrates one shared state vector with
    # explicit RK45, whose stability limit is ~2.8/|lambda|, so this one
    # occupancy component bounds the step for the whole system at ~2e-5 min:
    # of order a million steps for a 10-minute window. Measured: a
    # noradrenaline-dosed 10-minute run finishes in ~3 s under the
    # agonist-state set and did not finish in 240 s under this one.
    whole_cell = by_name(noradrenaline.ligands_alpha2(affinity_state="whole-cell"))
    koff = whole_cell["noradrenaline"].koff
    assert abs(koff - 129193.67) < 0.1

    stable_step = 2.8 / koff
    simulation_max_step = 0.1  # engine.Simulation.run's max_step
    assert stable_step < 1e-4
    assert T_SPAN[1] / stable_step > 4.0e5
    # ...against the default set, which is 4 orders of magnitude cheaper.
    agonist_state = by_name(noradrenaline.ligands_alpha2())
    assert 2.8 / agonist_state["noradrenaline"].koff > 100.0 * stable_step
    # The engine's own step cap is not what binds here; the ligand is.
    assert stable_step < simulation_max_step


def test_the_illustrative_pk_constants_are_shared_rather_than_invented():
    # One noradrenaline dose is applied to all THREE systems by name, so all
    # three must clear it at the same rate or the model describes one
    # molecule with three elimination rate constants.
    for models in (
        noradrenaline.pk_models_alpha1(),
        noradrenaline.pk_models_alpha2(),
        noradrenaline.pk_models_beta(),
    ):
        assert models[0].ke == noradrenaline.NORADRENALINE_KE
        for model in models[1:]:
            assert model.ke == noradrenaline.TOOL_COMPOUND_KE

    # The transmitter proxy is deliberately the same number the other two
    # transmitter modules use, and the tool-compound ke is deliberately the
    # same number P5.1 and P5.2 gave their tool compounds. Asserted so they
    # cannot drift apart.
    assert noradrenaline.NORADRENALINE_KE == dopamine.DOPAMINE_KE
    assert noradrenaline.NORADRENALINE_KE == serotonin.SEROTONIN_KE
    assert noradrenaline.TOOL_COMPOUND_KE == serotonin.TOOL_COMPOUND_KE
    assert noradrenaline.TOOL_COMPOUND_KE == dopamine.TOOL_COMPOUND_KE


# ---------------------------------------------------------------------------
# The circuit layer: does it obey its own three stated rules?
# ---------------------------------------------------------------------------


def test_the_network_is_three_nodes_two_edges_and_no_lc_to_cortex_edge():
    network = noradrenaline.locus_coeruleus_cortical_network()
    idx = network.index
    for name in ("LC_NA", "PFC_Pyramidal", "PFC_GABA"):
        assert name in idx

    W = np.asarray(network.connectivity)
    assert W[idx["PFC_Pyramidal"], idx["PFC_GABA"]] == -0.9
    assert W[idx["PFC_GABA"], idx["PFC_Pyramidal"]] == 0.8
    assert int(np.count_nonzero(W)) == 2

    # The missing edge is the point: the LC's influence on a cortical
    # pyramidal cell is receptor-mediated and of both signs at once
    # (alpha1 and beta excitatory, alpha2 inhibitory), so representing it as
    # one signed weight would double-count the receptor layer. Closing the
    # loop is P6.1.
    assert W[idx["PFC_Pyramidal"], idx["LC_NA"]] == 0.0
    assert not np.any(W[:, idx["LC_NA"]])
    assert not np.any(W[idx["LC_NA"], :])

    # The cortical loop is P5.1's, unchanged, because it stands for the same
    # microcircuit; asserted as equality of numbers so a future edit to
    # either module has to be deliberate.
    raphe = serotonin.raphe_cortical_network()
    r_idx = raphe.index
    assert (
        np.asarray(raphe.connectivity)[r_idx["PFC_Pyramidal"], r_idx["PFC_GABA"]]
        == W[idx["PFC_Pyramidal"], idx["PFC_GABA"]]
    )
    assert (
        np.asarray(raphe.connectivity)[r_idx["PFC_GABA"], r_idx["PFC_Pyramidal"]]
        == W[idx["PFC_GABA"], idx["PFC_Pyramidal"]]
    )
    # ...and so is the readout on the shared cortical node.
    assert (
        noradrenaline.cortical_output_readout().weights
        == serotonin.cortical_output_readout().weights
    )


def test_rule_one_rest_is_exactly_the_maximum_sensitivity_point():
    # Rule 1: baseline_input puts every node at rate 0.5 at rest, the
    # maximum-sensitivity point stability.MAX_SENSITIVITY_RATE derives.
    # Given the two loop weights that determines all three values, so
    # stability.balanced_baselines must reproduce them exactly.
    network = noradrenaline.locus_coeruleus_cortical_network()
    balanced = stability.balanced_baselines(
        network, target_rates=stability.MAX_SENSITIVITY_RATE
    )
    for node, rebuilt in zip(network.nodes, balanced.nodes):
        assert abs(node.baseline_input - rebuilt.baseline_input) < 1e-12

    rates, residual = stability.steady_state(network)
    assert residual < 1e-6
    assert np.allclose(rates, 0.5, atol=1e-6)
    assert np.allclose(noradrenaline.rest_rates(network), rates, atol=1e-6)


def test_rule_two_the_contraction_certificate_holds_unlike_p5_2s_network():
    # Rule 2: sigma_max(diag(g/4) W) < 1, P3.4's global-uniqueness
    # certificate, so this network has exactly one fixed point and every
    # trajectory converges to it whatever the taus. With gain 4 everywhere
    # that is max(0.9, 0.8) = 0.9.
    network = noradrenaline.locus_coeruleus_cortical_network()
    assessment = stability.assess(network)
    assert abs(assessment.contraction - 0.9) < 1e-9
    assert assessment.globally_certified
    assert assessment.converged
    assert assessment.unique
    assert assessment.locally_stable
    assert assessment.saturation.ok
    assert assessment.ok

    # Worth asserting alongside P5.2, where the certificate FAILS
    # (kappa 1.414) and acyclicity does the work instead. The two networks
    # in this codebase are certified by different arguments, and that is a
    # fact about them rather than an accident.
    assert not stability.assess(dopamine.cortico_basal_ganglia_network()).globally_certified


def test_rule_three_is_a_per_node_budget_and_it_rejects_the_per_receptor_one():
    # Rule 3: no node saturates under a saturating dose of a full agonist at
    # every receptor it carries. `PFC_Pyramidal` carries TWO excitatory
    # receptors, so P5.1's per-receptor unit rule would put +2.0 of drive on
    # it, and that rails it. The unit budget is therefore per NODE: two
    # gains of +0.5 summing to the one unit P5.1 allots.
    assert noradrenaline.ALPHA1_MECHANISM_GAIN == 0.5
    assert noradrenaline.BETA_MECHANISM_GAIN == 0.5
    assert (
        noradrenaline.ALPHA1_MECHANISM_GAIN + noradrenaline.BETA_MECHANISM_GAIN == 1.0
    )
    # The LC node carries one receptor, so it keeps the unit magnitude.
    assert noradrenaline.ALPHA2_MECHANISM_GAIN == -1.0

    network = noradrenaline.locus_coeruleus_cortical_network()
    idx = network.index
    # Full occupancy by a full agonist at both cortical receptors, which is
    # what the two gains sum to.
    for total_drive, must_be_ok in ((1.0, True), (2.0, False)):
        drive = network.baseline_drive().copy()
        drive[idx["PFC_Pyramidal"]] += total_drive
        rates, residual = stability.steady_state(network, drive=drive)
        assert residual < 1e-6
        report = stability.check_saturation(network, rates, drive=drive)
        assert report.ok is must_be_ok
        if must_be_ok:
            assert report.min_sensitivity > 0.19
        else:
            # The counterfactual, which is the reason the number is 0.5: at
            # the per-receptor unit rule the cortical node's sigmoid
            # sensitivity falls to 0.0044, below SATURATION_FLOOR (0.01),
            # and a railed node stops responding to the other receptor.
            assert report.min_sensitivity < stability.SATURATION_FLOOR
            assert abs(report.min_sensitivity - 0.00442) < 0.0002

    # ...and the same rule at the LC node, where the single unit gain is
    # what a saturating alpha2 agonist delivers.
    drive = network.baseline_drive().copy()
    drive[idx["LC_NA"]] += noradrenaline.ALPHA2_MECHANISM_GAIN
    rates, residual = stability.steady_state(network, drive=drive)
    assert residual < 1e-6
    assert stability.check_saturation(network, rates, drive=drive).ok
    assert rates[idx["LC_NA"]] < 0.03  # the autoreceptor nearly silences it


def test_a_drug_free_run_sits_exactly_at_rest(sim):
    _result, rates = traces(sim)
    for node in ("LC_NA", "PFC_Pyramidal", "PFC_GABA"):
        assert np.allclose(rates[node], 0.5, atol=1e-6)


# ---------------------------------------------------------------------------
# The pharmacology: one transmitter, three receptors, opposite signs
# ---------------------------------------------------------------------------


def test_noradrenaline_excites_cortex_and_suppresses_the_locus_coeruleus(sim):
    # THE result this task exists to produce: ONE transmitter dose moving
    # two nodes in OPPOSITE directions, because the three receptors it binds
    # have different G proteins. alpha1 (Gq, [McCormick1991]) and beta1 (Gs,
    # [Madison1986]) add drive to the cortical pyramidal node; alpha2 (Gi,
    # [Cedarbaum1977]) removes drive from the LC node that carries the
    # autoreceptor.
    _result, rates = traces(sim, noradrenaline=NORADRENALINE_DOSE_nM)

    assert rates["PFC_Pyramidal"].max() > 0.5
    assert rates["LC_NA"].min() < 0.5
    # Not a rounding-error separation: the drug-free trace is flat to 1e-6
    # (test_a_drug_free_run_sits_exactly_at_rest), so these are 4 and 5
    # orders of magnitude above the integrator's own noise.
    assert rates["PFC_Pyramidal"].max() - 0.5 > 0.01
    assert 0.5 - rates["LC_NA"].min() > 0.1

    # The cortical response is much smaller than the peak occupancy would
    # support, and that is the transmitter-clearance proxy rather than the
    # pharmacology: NORADRENALINE_KE = 5.0 min^-1 is a half-life of 8.3 s,
    # so the drive is gone before a tau = 1.5 min node can follow it. The
    # steady state the same occupancy held constant would reach is computed
    # here to show the size of that gap, and it is why P6.2's tonic release
    # matters for anything quantitative.
    network = sim.network
    idx = network.index
    alpha1_occ = equilibrium_occupancy(
        by_name(noradrenaline.ligands_alpha1())["noradrenaline"],
        NORADRENALINE_DOSE_nM,
    )
    beta_occ = equilibrium_occupancy(
        by_name(noradrenaline.ligands_beta())["noradrenaline"],
        NORADRENALINE_DOSE_nM,
    )
    drive = network.baseline_drive().copy()
    drive[idx["PFC_Pyramidal"]] += (
        noradrenaline.ALPHA1_MECHANISM_GAIN * alpha1_occ
        + noradrenaline.BETA_MECHANISM_GAIN * beta_occ
    )
    sustained, residual = stability.steady_state(network, drive=drive)
    assert residual < 1e-6
    assert sustained[idx["PFC_Pyramidal"]] > rates["PFC_Pyramidal"].max()
    assert sustained[idx["PFC_Pyramidal"]] > 0.7


def test_each_antagonist_removes_only_its_own_receptors_contribution(sim):
    # alpha1 and beta land on the SAME node with the same sign, so the only
    # way to show they are two receptors and not one is to remove them one
    # at a time. prazosin (Kd 0.90 nM) takes the alpha1 pool, propranolol
    # (Kd 0.71 nM) takes the beta1 pool, and each removes part of the
    # cortical response.
    _r, plain = traces(sim, noradrenaline=NORADRENALINE_DOSE_nM)
    _r, no_alpha1 = traces(
        sim, noradrenaline=NORADRENALINE_DOSE_nM, prazosin=PRAZOSIN_DOSE_nM
    )
    _r, no_beta = traces(
        sim, noradrenaline=NORADRENALINE_DOSE_nM, propranolol=PROPRANOLOL_DOSE_nM
    )
    _r, neither = traces(
        sim,
        noradrenaline=NORADRENALINE_DOSE_nM,
        prazosin=PRAZOSIN_DOSE_nM,
        propranolol=PROPRANOLOL_DOSE_nM,
    )

    # Blocking one receptor leaves the OTHER receptor's contribution
    # standing, so the part attributable to a receptor is the peak minus
    # what survives blocking it: prazosin's run measures the beta part and
    # propranolol's run measures the alpha1 part.
    peak = plain["PFC_Pyramidal"].max() - 0.5
    alpha1_part = peak - (no_alpha1["PFC_Pyramidal"].max() - 0.5)
    beta_part = peak - (no_beta["PFC_Pyramidal"].max() - 0.5)
    both = peak - (neither["PFC_Pyramidal"].max() - 0.5)

    # Each antagonist removes a real part of the response...
    assert alpha1_part > 0.0
    assert beta_part > 0.0
    # ...the beta part is the larger one, which is a PREDICTION and not a
    # parameter: the two gains are equal by construction (rule 3), so the
    # asymmetry comes only from noradrenaline binding beta1 8.5-fold more
    # tightly than alpha1A ([Baker2010] vs [Proudman2021]).
    assert beta_part > alpha1_part
    # ...and together they remove nearly all of it, because the node carries
    # no other drug target.
    assert both > 0.8 * peak

    # Neither antagonist touches the LC node, which carries neither
    # receptor. 1e-3 rather than 0 is RK45 step-size selection, the same
    # residual P5.1's open-loop test documents: one step is chosen for the
    # whole shared state vector, so adding a ligand perturbs the last digits
    # of every other trace.
    assert np.abs(neither["LC_NA"] - plain["LC_NA"]).max() < 1e-3


def test_clonidine_suppresses_the_locus_coeruleus_and_yohimbine_reverses_it(sim):
    # [Cedarbaum1977]: clonidine depresses rat locus-coeruleus firing and
    # alpha-antagonists including yohimbine block the effect. Here clonidine
    # occupies the autoreceptor pool with efficacy 0.32 [Jasper1998], the
    # Gi-coupled gain is negative, and the LC node falls; yohimbine (Kd
    # 3.31 nM, efficacy 0) competes for the same pool and gives it back.
    _r, clonidine_only = traces(sim, clonidine=CLONIDINE_DOSE_nM)
    _r, reversed_low = traces(
        sim, clonidine=CLONIDINE_DOSE_nM, yohimbine=YOHIMBINE_DOSE_nM
    )
    _r, reversed_high = traces(
        sim, clonidine=CLONIDINE_DOSE_nM, yohimbine=5.0 * YOHIMBINE_DOSE_nM
    )

    assert clonidine_only["LC_NA"].min() < 0.3
    # Reversal is PARTIAL at equal doses and greater at a larger one, which
    # is what competition for one pool by two ligands 1.8-fold apart in
    # affinity (clonidine 5.9 nM, yohimbine 3.31 nM) has to look like. A
    # test that asserted full reversal at an equimolar dose would be
    # asserting the wrong pharmacology.
    assert (
        clonidine_only["LC_NA"].min()
        < reversed_low["LC_NA"].min()
        < reversed_high["LC_NA"].min()
    )
    assert reversed_high["LC_NA"].min() > 0.45

    # Clonidine leaves the cortical node alone: it is an alpha2 ligand and
    # the cortical node carries alpha1 and beta1 only.
    assert np.allclose(clonidine_only["PFC_Pyramidal"], 0.5, atol=1e-6)


def test_clonidines_partial_agonism_is_visible_as_a_partial_effect(sim):
    # Clonidine is a weak partial agonist (0.32 of noradrenaline
    # [Jasper1998]), so at FULL occupancy of the autoreceptor pool it can
    # only deliver 0.32 of the drive a full agonist would. Checked against
    # the algebraic steady state the same occupancy with efficacy 1.0 would
    # produce, so the claim is about efficacy and not about dose.
    network = sim.network
    idx = network.index
    ligands = noradrenaline.ligands_alpha2()
    occupancy = equilibrium_occupancy(by_name(ligands)["clonidine"], CLONIDINE_DOSE_nM)
    assert occupancy > 0.99  # the pool really is full at this dose

    partial = network.baseline_drive().copy()
    partial[idx["LC_NA"]] += (
        noradrenaline.ALPHA2_MECHANISM_GAIN
        * occupancy
        * noradrenaline.ALPHA2_EFFICACY_CLONIDINE
    )
    full = network.baseline_drive().copy()
    full[idx["LC_NA"]] += noradrenaline.ALPHA2_MECHANISM_GAIN * occupancy

    partial_rates, _ = stability.steady_state(network, drive=partial)
    full_rates, _ = stability.steady_state(network, drive=full)
    assert full_rates[idx["LC_NA"]] < partial_rates[idx["LC_NA"]] < 0.5

    # And the dynamic run lands on the partial value, not the full one.
    _r, rates = traces(sim, clonidine=CLONIDINE_DOSE_nM)
    assert abs(rates["LC_NA"][-1] - partial_rates[idx["LC_NA"]]) < 0.02


def test_clonidine_is_a_functional_antagonist_against_the_transmitter(sim):
    # A result predicted backwards on the way in and worth keeping for that
    # reason. Clonidine SUPPRESSES the LC node when given alone (the test
    # above) and RAISES it when given against a noradrenaline background,
    # because a partial agonist competing with a full agonist for one pool
    # is a functional ANTAGONIST. Both halves are the same two published
    # numbers: clonidine binds the alpha2 pool 4.6-fold more tightly than
    # noradrenaline in the default affinity state (5.9 nM [Salama1982]
    # against 27 nM [UPrichard1978]) and activates it 0.32 as well
    # [Jasper1998], so it displaces occupancy worth 1.0 of activation and
    # replaces it with occupancy worth 0.32.
    ligands = noradrenaline.ligands_alpha2()
    by_ligand = dict(zip([lig.name for lig in ligands], ligands))
    assert kd(by_ligand["clonidine"]) < kd(by_ligand["noradrenaline"])
    assert by_ligand["clonidine"].efficacy < by_ligand["noradrenaline"].efficacy

    # At the algebraic competitive fixed point, adding clonidine to
    # noradrenaline LOWERS the pool's total activation.
    names = [lig.name for lig in ligands]
    without = competitive_equilibrium(
        ligands, [NORADRENALINE_DOSE_nM if n == "noradrenaline" else 0.0 for n in names]
    )
    with_clonidine = competitive_equilibrium(
        ligands,
        [
            NORADRENALINE_DOSE_nM
            if n == "noradrenaline"
            else (CLONIDINE_DOSE_nM if n == "clonidine" else 0.0)
            for n in names
        ],
    )
    site = noradrenaline.receptor_site_alpha2()
    assert site.activation(with_clonidine) < site.activation(without)

    # ...and the integrated run agrees: the LC node sits HIGHER with
    # clonidine on board than with noradrenaline alone.
    _r, plain = traces(sim, noradrenaline=NORADRENALINE_DOSE_nM)
    _r, plus_clonidine = traces(
        sim, noradrenaline=NORADRENALINE_DOSE_nM, clonidine=CLONIDINE_DOSE_nM
    )
    assert plus_clonidine["LC_NA"].min() > plain["LC_NA"].min()

    # This is state-dependent, and the reason is the affinity spread rather
    # than anything hand-set: in the whole-cell set noradrenaline barely
    # occupies the alpha2 pool at this dose, so there is nothing for
    # clonidine to displace and it suppresses instead. Asserted
    # algebraically, because that set cannot be integrated.
    whole_cell = noradrenaline.ligands_alpha2(affinity_state="whole-cell")
    wc_names = [lig.name for lig in whole_cell]
    wc_without = competitive_equilibrium(
        whole_cell,
        [NORADRENALINE_DOSE_nM if n == "noradrenaline" else 0.0 for n in wc_names],
    )
    wc_with = competitive_equilibrium(
        whole_cell,
        [
            NORADRENALINE_DOSE_nM
            if n == "noradrenaline"
            else (CLONIDINE_DOSE_nM if n == "clonidine" else 0.0)
            for n in wc_names
        ],
    )
    wc_site = noradrenaline.receptor_site_alpha2(affinity_state="whole-cell")
    assert wc_site.activation(wc_with) > wc_site.activation(wc_without)


def test_every_antagonist_alone_is_inert_until_phase_6(sim):
    # With no endogenous noradrenergic tone an antagonist has nothing to
    # displace, so it does nothing -- the same gap naloxone, haloperidol,
    # WAY-100635 and SCH-23390 have in every other scenario here. Real
    # prazosin, real yohimbine and real propranolol all have effects given
    # alone, because a real locus coeruleus fires tonically. Supplying that
    # tone is P6.2, and the day it lands this expectation is already written
    # down: these three assertions must FAIL then.
    for doses in (
        {"prazosin": PRAZOSIN_DOSE_nM},
        {"propranolol": PROPRANOLOL_DOSE_nM},
        {"yohimbine": YOHIMBINE_DOSE_nM},
    ):
        _r, rates = traces(sim, **doses)
        for node in ("LC_NA", "PFC_Pyramidal", "PFC_GABA"):
            assert np.allclose(rates[node], 0.5, atol=1e-6)


def test_the_autoreceptor_is_an_open_loop_until_phase_6(sim):
    # P5.1 asserts this gap in general form for the 5-HT1A autoreceptor;
    # it is re-asserted here in the form specific to the LC, because THIS
    # is the autoreceptor whose whole clinical point is the missing half.
    # An alpha2 agonist activates the somatodendritic autoreceptor, LC
    # firing falls, and in a real brain LESS NORADRENALINE IS RELEASED
    # ACROSS THE ENTIRE FOREBRAIN ([Callado2000], [Berridge2003]) -- which
    # is how clonidine sedates and lowers blood pressure. This model can
    # express the first half and not the second, because `Simulation`'s
    # ligand concentrations come from the PK layer and no circuit rate has
    # a path back into them.
    #
    # So: clonidine moves the LC node, and the cortical alpha1 and beta1
    # pools see exactly the noradrenaline they would have seen anyway.
    # Closing the loop in P6.1/P6.2 must break this test.
    result_plain, plain = traces(sim, noradrenaline=NORADRENALINE_DOSE_nM)
    result_clonidine, with_clonidine = traces(
        sim, noradrenaline=NORADRENALINE_DOSE_nM, clonidine=CLONIDINE_DOSE_nM
    )

    # The first half, which the model does have. Only the SIZE of the change
    # is asserted here and not its sign, because against a noradrenaline
    # background clonidine's sign is the partial-agonist one and that is
    # tested on its own below.
    assert abs(with_clonidine["LC_NA"].min() - plain["LC_NA"].min()) > 0.01

    # The second half, which it does not: same cortical noradrenaline, same
    # cortical occupancy, same cortical firing.
    #
    # The residual is RK45 step-size selection over the shared state vector,
    # the same effect P5.1's version of this test documents, and it is
    # LARGER here than P5.1's ~2e-6 for a stated reason rather than a
    # mysterious one: this module's fastest occupancy component runs at
    # 743 min^-1 (noradrenaline at alpha1, derivation 3) against ~1 min^-1
    # in the serotonin module, so the same step-placement difference samples
    # a much steeper trace. The measured worst-case relative difference is
    # 8.9e-4 in the occupancy traces and below 1e-4 in the concentration
    # traces. The tolerance is therefore `solve_ivp`'s own default rtol of
    # 1e-3, which is the accuracy any single sample is guaranteed to -- a
    # bound taken from the integrator rather than fitted to the number, and
    # 3 to 4 orders of magnitude below the 0.1-0.3 changes the LC node
    # shows.
    solver_rtol = 1e-3  # scipy.integrate.solve_ivp's default

    def relative_max_difference(a, b):
        scale = max(np.abs(a).max(), 1e-12)
        return np.abs(a - b).max() / scale

    alpha1_row, beta_row = 0, 4  # noradrenaline in the alpha1 and beta sites
    for row in (alpha1_row, beta_row):
        assert (
            relative_max_difference(
                result_plain["concentrations"][row],
                result_clonidine["concentrations"][row],
            )
            < 1e-4
        )
        assert (
            relative_max_difference(
                result_plain["occupancy"][row], result_clonidine["occupancy"][row]
            )
            < solver_rtol
        )
    assert (
        relative_max_difference(plain["PFC_Pyramidal"], with_clonidine["PFC_Pyramidal"])
        < solver_rtol
    )

    # And the structural reason, which is what actually makes the loop open
    # and is not a tolerance at all: every ligand's concentration is driven
    # by a PK model whose right-hand side is a function of its own state
    # only, so no circuit rate can reach it. P6.1 is exactly the change that
    # replaces one of these with a release term that depends on `LC_NA`.
    for system in sim.receptor_systems:
        for model in system.pk_models:
            assert isinstance(model, OneCompartmentPK)
            assert model.concentration_rhs([1.0])[0] == -model.ke * 1.0


# ---------------------------------------------------------------------------
# The same results under the other affinity state, and the other propranolol
# affinity
# ---------------------------------------------------------------------------


def test_the_clonidine_result_holds_under_both_alpha2_affinity_states():
    # The clonidine/yohimbine results do not dose noradrenaline, so the
    # whole-cell set can be integrated for them and the direction of every
    # claim above can be checked under both states. What changes is the
    # MAGNITUDE of the reversal, and by a knowable amount: yohimbine is
    # 1.8-fold tighter than clonidine in the agonist state (3.31 vs 5.9 nM)
    # and 58-fold tighter in the whole-cell one (3.31 vs 190.5 nM), so
    # equimolar yohimbine reverses much more completely there.
    reversal = {}
    for state in noradrenaline.AFFINITY_STATES:
        simulation = build_simulation(affinity_state=state)
        _r, clonidine_only = traces(simulation, clonidine=CLONIDINE_DOSE_nM)
        _r, reversed_ = traces(
            simulation, clonidine=CLONIDINE_DOSE_nM, yohimbine=YOHIMBINE_DOSE_nM
        )
        assert clonidine_only["LC_NA"].min() < 0.35
        assert reversed_["LC_NA"].min() > clonidine_only["LC_NA"].min()
        reversal[state] = reversed_["LC_NA"].min() - clonidine_only["LC_NA"].min()
    assert reversal["whole-cell"] > reversal["agonist-state"]


def test_noradrenaline_at_the_whole_cell_alpha2_affinity_barely_engages_it():
    # The whole-cell counterpart of
    # test_noradrenaline_excites_cortex_and_suppresses_the_locus_coeruleus,
    # asserted at the algebraic fixed point because that set cannot be
    # integrated. Under the matched panel's own number the same dose that
    # occupies 62% of the beta1 pool occupies 1.1% of the alpha2 pool, so
    # the LC would hardly move at all -- the opposite of the default set's
    # result, and the sharpest statement of what the affinity-state choice
    # costs.
    whole_cell = by_name(noradrenaline.ligands_alpha2(affinity_state="whole-cell"))
    agonist_state = by_name(noradrenaline.ligands_alpha2())
    whole_cell_occ = equilibrium_occupancy(
        whole_cell["noradrenaline"], NORADRENALINE_DOSE_nM
    )
    agonist_state_occ = equilibrium_occupancy(
        agonist_state["noradrenaline"], NORADRENALINE_DOSE_nM
    )
    assert abs(whole_cell_occ - 0.0110) < 0.0005
    assert agonist_state_occ > 0.99

    # Turned into the node rate each would produce, so the comparison is on
    # the quantity the scenario reports rather than on occupancy alone.
    network = noradrenaline.locus_coeruleus_cortical_network()
    idx = network.index
    lc_rates = {}
    for label, occupancy in (
        ("whole-cell", whole_cell_occ),
        ("agonist-state", agonist_state_occ),
    ):
        drive = network.baseline_drive().copy()
        drive[idx["LC_NA"]] += noradrenaline.ALPHA2_MECHANISM_GAIN * occupancy
        rates, residual = stability.steady_state(network, drive=drive)
        assert residual < 1e-6
        lc_rates[label] = rates[idx["LC_NA"]]
    assert lc_rates["whole-cell"] > 0.48
    assert lc_rates["agonist-state"] < 0.03


def test_no_beta_result_turns_on_which_propranolol_affinity_is_used():
    # 9.7-fold apart ([Sykes2022]'s kinetic 0.711 nM against [Baker2005]'s
    # whole-cell 6.92 nM), and the module uses the kinetic one because its
    # rate constants are measured. This asserts the claim that goes with
    # that choice: propranolol is 2000-fold tighter than noradrenaline at
    # beta1 under the kinetic number and 263-fold tighter under the
    # whole-cell one, so it wins the pool either way and the reversal is the
    # same result.
    kinetic = by_name(noradrenaline.ligands_beta())["propranolol"]
    whole_cell = by_name(
        noradrenaline.ligands_beta(propranolol_affinity="whole-cell")
    )["propranolol"]
    assert abs(kd(whole_cell) / kd(kinetic) - 9.73) < 0.05
    for propranolol in (kinetic, whole_cell):
        assert noradrenaline.BETA_KD_NORADRENALINE / kd(propranolol) > 250.0

    with pytest.raises(ValueError):
        noradrenaline.ligands_beta(propranolol_affinity="the-other-one")

    # And the integrated result, under both.
    peaks = {}
    for affinity in noradrenaline.PROPRANOLOL_AFFINITIES:
        simulation = build_simulation(propranolol_affinity=affinity)
        _r, plain = traces(simulation, noradrenaline=NORADRENALINE_DOSE_nM)
        _r, blocked = traces(
            simulation,
            noradrenaline=NORADRENALINE_DOSE_nM,
            propranolol=PROPRANOLOL_DOSE_nM,
        )
        assert blocked["PFC_Pyramidal"].max() < plain["PFC_Pyramidal"].max()
        peaks[affinity] = blocked["PFC_Pyramidal"].max()
    # The tighter set blocks slightly more, which is the only difference.
    assert peaks["kinetic"] <= peaks["whole-cell"]


def test_the_three_pools_compete_only_within_themselves():
    # Three sites, three separate pools, and one transmitter present in all
    # three by name. The competitive fixed point is computed per site, so a
    # cross-site leak would show up as an occupancy that depends on a ligand
    # the site does not carry. Checked algebraically across the whole set at
    # once, which is cheap and exact.
    concentrations = {
        "noradrenaline": NORADRENALINE_DOSE_nM,
        "prazosin": PRAZOSIN_DOSE_nM,
        "clonidine": CLONIDINE_DOSE_nM,
        "yohimbine": YOHIMBINE_DOSE_nM,
        "propranolol": PROPRANOLOL_DOSE_nM,
    }
    for ligands in (
        noradrenaline.ligands_alpha1(),
        noradrenaline.ligands_alpha2(),
        noradrenaline.ligands_beta(),
    ):
        doses = [concentrations[ligand.name] for ligand in ligands]
        occupancies = competitive_equilibrium(ligands, doses)
        assert sum(occupancies) < 1.0 + 1e-12
        # The antagonist wins its own pool at these doses, which is what
        # every reversal result above rests on.
        antagonists = [
            occ
            for occ, ligand in zip(occupancies, ligands)
            if ligand.efficacy == 0.0
        ]
        if antagonists:
            assert max(antagonists) > 0.5
