"""Tests for the calibrated D2/dopamine/haloperidol parameter set
(scenarios.py, "Calibrated D2 receptor + dopamine/haloperidol PK" section).
See that section's docstring/comments for full source citations; each test
below cites the specific literature figure it checks against.
"""

import numpy as np

from neurocascade import scenarios
from neurocascade.engine import Simulation
from neurocascade.receptor import ReceptorSite

HALOPERIDOL_MW = 375.86  # g/mol


def ng_ml_to_nM(ng_per_ml, mw=HALOPERIDOL_MW):
    return (ng_per_ml * 1e-6 / mw) * 1e9


def test_calibrated_kon_koff_reproduce_kd_at_equilibrium():
    # Sanity check on the reported kinetics themselves: for a single ligand
    # at equilibrium, occ = kon*C / (kon*C + koff) = C / (C + Kd), where
    # Kd = koff/kon. At C == Kd that must equal 0.5 for both ligands, or
    # there is a units/arithmetic error in how scenarios.py converted the
    # published rate constants.
    dopamine, haloperidol = scenarios.d2_dopamine_ligands_calibrated()

    for ligand in (dopamine, haloperidol):
        Kd = ligand.koff / ligand.kon
        site = ReceptorSite(name="site", ligands=[ligand])
        occ = np.array([0.0])
        dt = 0.01
        for _ in range(5000):
            occ = occ + dt * site.occupancy_rhs(occ, [Kd])
        assert abs(occ[0] - 0.5) < 1e-3


def test_calibrated_dopamine_koff_faster_than_haloperidol_koff():
    # Agren et al. 2021 (Int J Mol Sci, functional GIRK-channel kinetics):
    # dopamine koff = 0.197 s^-1 (11.82 min^-1). Sykes et al. 2017 (Nat
    # Commun, TR-FRET competition kinetics): haloperidol koff = 0.65 min^-1.
    # Dopamine's dissociation from D2 is genuinely far faster (~18x) than
    # haloperidol's, consistent with dopamine being the fast, low-affinity,
    # rapidly-dissociating endogenous agonist and haloperidol the much
    # slower-dissociating antagonist, so this part of the illustrative
    # placeholder's qualitative assumption holds up under real data.
    #
    # What does NOT hold up (see the long comment above
    # d2_dopamine_ligands_calibrated in scenarios.py for full citations): the
    # common pharmacology-folklore claim that haloperidol has an unusually
    # SLOW koff *relative to other antipsychotics*, and that this alone
    # mechanistically explains its EPS liability. Sykes et al. 2017's own
    # panel puts haloperidol's koff toward the FAST end even among typical
    # antipsychotics (fluphenazine, spiperone, nemonapride, (+)-butaclamol
    # all dissociate several- to 30-fold more slowly than haloperidol in the
    # same assay), and haloperidol's koff (0.65 min^-1) is actually faster
    # than the atypical risperidone's (0.43 min^-1) in that dataset. A
    # published critique of that paper (Zeberg & Sahlholm 2018) argues it is
    # haloperidol's unusually HIGH association rate (kon), not a slow koff,
    # that better explains its EPS liability. This codebase's two-ligand
    # scenario can only faithfully represent the dopamine-vs-haloperidol
    # comparison checked here, not the haloperidol-vs-other-antipsychotics
    # comparison the folklore claim is actually about.
    dopamine, haloperidol = scenarios.d2_dopamine_ligands_calibrated()
    assert dopamine.koff > haloperidol.koff


def test_calibrated_haloperidol_occupancy_near_pet_occupancy_targets():
    # Validation target: Kapur et al. 1997 (Psychopharmacology, PET with
    # [11C]-raclopride in patients on low-dose oral haloperidol) fit plasma
    # haloperidol level to striatal D2 occupancy as a saturating hyperbola:
    # 50% occupancy at a plasma level of 0.51 ng/mL, 80% occupancy at
    # 2.0 ng/mL. Converting those plasma levels to nM (haloperidol MW
    # 375.86 g/mol) and feeding them into this codebase's simple two-state
    # occupancy equation (no plasma-protein-binding or brain/plasma
    # partitioning submodel, the same simplification flagged for the
    # fentanyl/Moss et al. 2020 comparison in the mu-opioid calibrated
    # section) predicts occupancy that is directionally right: clearly
    # partial-to-high at the clinical ED50 concentration and clearly
    # near-saturating at the clinical 80%-occupancy concentration, but
    # running somewhat HIGHER than the clinical percentages themselves. That
    # over-prediction is expected and explainable: haloperidol is ~89-93%
    # bound to plasma protein in vivo, so total plasma concentration
    # overstates the pharmacologically-active free concentration this
    # model implicitly assumes is fully available to bind; this is an
    # order-of-magnitude/directional check, not an exact-percentage match.
    _dopamine, haloperidol = scenarios.d2_dopamine_ligands_calibrated()
    site = ReceptorSite(name="site", ligands=[haloperidol])

    c_ed50_nM = ng_ml_to_nM(0.51)  # Kapur et al. 1997: 50% occupancy target
    c_80pct_nM = ng_ml_to_nM(2.0)  # Kapur et al. 1997: 80% occupancy target

    def equilibrium_occupancy(C):
        occ = np.array([0.0])
        dt = 0.01
        for _ in range(5000):
            occ = occ + dt * site.occupancy_rhs(occ, [C])
        return occ[0]

    occ_ed50 = equilibrium_occupancy(c_ed50_nM)
    occ_80pct = equilibrium_occupancy(c_80pct_nM)

    assert 0.5 < occ_ed50 < 1.0
    assert occ_80pct > 0.9
    assert occ_80pct > occ_ed50  # monotonic in concentration, as it must be


def build_calibrated_sim():
    return Simulation(
        pk_models=scenarios.d2_pk_models_calibrated(),
        receptor_site=scenarios.d2_receptor_site_calibrated(),
        network=scenarios.basal_ganglia_indirect_pathway_network(),
        coupling=scenarios.d2_coupling(),
    )


def test_calibrated_agonist_raises_locomotor_activity_above_baseline():
    # With real receptor/PK numbers substituted for the illustrative
    # placeholders, the qualitative v0 mechanism (a D2 agonist disinhibits
    # the indirect pathway and raises the locomotor-activity readout above
    # baseline) must still hold; the circuit wiring itself (still
    # illustrative) is unchanged. Dopamine's calibrated PK clears in
    # seconds (see d2_pk_models_calibrated's ke=5.0 proxy), so this uses a
    # short simulation window that actually resolves the transient, unlike
    # the mu-opioid calibrated tests' 60-minute window.
    sim = build_calibrated_sim()
    readout = scenarios.locomotor_activity_readout()
    t_eval = np.linspace(0, 10, 400)

    baseline = sim.run(dose_concentrations=[0.0, 0.0], t_span=(0, 10), t_eval=t_eval)
    agonist = sim.run(dose_concentrations=[100.0, 0.0], t_span=(0, 10), t_eval=t_eval)

    baseline_scores = np.array(
        [readout.score(sim.network, baseline["rates"][:, i]) for i in range(len(t_eval))]
    )
    agonist_scores = np.array(
        [readout.score(sim.network, agonist["rates"][:, i]) for i in range(len(t_eval))]
    )
    assert agonist_scores.max() > baseline_scores.max()


def test_calibrated_antagonist_alone_stays_near_baseline():
    # Haloperidol has nothing to displace without endogenous dopamine
    # present in this bolus-only model, so given alone it should track
    # baseline closely, the same check test_d2_scenario.py makes for the
    # illustrative parameter set.
    sim = build_calibrated_sim()
    readout = scenarios.locomotor_activity_readout()
    t_eval = np.linspace(0, 10, 400)

    baseline = sim.run(dose_concentrations=[0.0, 0.0], t_span=(0, 10), t_eval=t_eval)
    antagonist = sim.run(dose_concentrations=[0.0, 100.0], t_span=(0, 10), t_eval=t_eval)

    baseline_final = readout.score(sim.network, baseline["rates"][:, -1])
    antagonist_final = readout.score(sim.network, antagonist["rates"][:, -1])
    assert abs(antagonist_final - baseline_final) < 1e-2


def test_calibrated_haloperidol_blunts_dopamine_agonist_effect():
    # Haloperidol given alongside dopamine, at twice dopamine's molar
    # concentration (and >600x haloperidol's own Kd), should reduce the
    # agonist-driven peak locomotor-activity score relative to dopamine
    # alone: haloperidol's real affinity is markedly tighter than
    # dopamine's (Kd ~0.31 nM vs ~2.03 nM, from koff/kon above), so at
    # this concentration it should win a meaningful share of the shared
    # receptor pool away from dopamine, consistent with its real-world
    # action as a D2 antagonist/EPS-causing agent when dopamine tone is
    # present. (A much larger excess, e.g. 3000 nM, gives the same
    # qualitative result but makes the ODE system's rate constants stiff
    # enough that RK45 becomes far slower to integrate; 200 nM keeps the
    # same directional check cheap to run.)
    sim = build_calibrated_sim()
    readout = scenarios.locomotor_activity_readout()
    t_eval = np.linspace(0, 10, 400)

    agonist_alone = sim.run(
        dose_concentrations=[100.0, 0.0], t_span=(0, 10), t_eval=t_eval
    )
    blocked = sim.run(dose_concentrations=[100.0, 200.0], t_span=(0, 10), t_eval=t_eval)

    agonist_scores = np.array(
        [readout.score(sim.network, agonist_alone["rates"][:, i]) for i in range(len(t_eval))]
    )
    blocked_scores = np.array(
        [readout.score(sim.network, blocked["rates"][:, i]) for i in range(len(t_eval))]
    )
    assert blocked_scores.max() < agonist_scores.max()
