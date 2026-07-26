"""Tests for the calibrated mu-opioid/fentanyl/naloxone parameter set
(scenarios.py, "Calibrated mu-opioid receptor + fentanyl/naloxone PK"
section). See that section's docstring/comments for full source citations;
each test below cites the specific literature figure it checks against.
"""

import numpy as np

from neurocascade import scenarios
from neurocascade.engine import Simulation
from neurocascade.receptor import ReceptorSite


def test_calibrated_kon_koff_reproduce_ki_at_equilibrium():
    # Sanity check on the kon = koff / Ki derivation itself: for a single
    # ligand at equilibrium, occ = kon*C / (kon*C + koff) = C / (C + Ki).
    # At C == Ki that must equal 0.5 for both ligands, or the derivation in
    # scenarios.py has a units/arithmetic error.
    fentanyl, naloxone = scenarios.mu_opioid_ligands_calibrated()
    Ki_fentanyl, Ki_naloxone = 1.346, 1.518  # Volpe et al. 2011, nM

    for ligand, Ki in [(fentanyl, Ki_fentanyl), (naloxone, Ki_naloxone)]:
        site = ReceptorSite(name="site", ligands=[ligand])
        occ = np.array([0.0])
        dt = 0.01
        for _ in range(200000):
            occ = occ + dt * site.occupancy_rhs(occ, [Ki])
        assert abs(occ[0] - 0.5) < 1e-3


def test_calibrated_fentanyl_koff_slower_than_naloxone_koff():
    # Mann et al. 2022 (FDA mechanistic model, in vitro binding-kinetics
    # fits): fentanyl koff = 0.004331 s^-1 (residence time ~160 s) is an
    # order of magnitude SLOWER than naloxone koff = 0.03959 s^-1
    # (residence time ~17.5 s), even though the two ligands have similar
    # affinity (Ki within ~15% of each other, Volpe et al. 2011). This is
    # the opposite of the illustrative placeholder scenario, which assumed
    # naloxone was the slower-koff ("stickier") ligand to make displacement
    # easy to reason about. Real naloxone reversal instead relies on being
    # given in large molar excess, not on higher receptor affinity or a
    # slower off-rate; see test_naloxone_reverses_calibrated_agonist_effect
    # below for confirmation that displacement still works under real
    # kinetics when dosed that way.
    fentanyl, naloxone = scenarios.mu_opioid_ligands_calibrated()
    assert fentanyl.koff < naloxone.koff


def test_calibrated_fentanyl_occupancy_near_saturating_at_overdose_concentration():
    # Validation target: Moss et al. 2020 (PLoS ONE, open-access QSP model
    # of naloxone-fentanyl competition at the mu receptor, using the same
    # fentanyl Ki = 1.35 nM as this codebase) report baseline (no naloxone)
    # mu-receptor occupancy of 97% at a fentanyl exposure of 50 ng/mL
    # (~148.6 nM, using fentanyl's molar mass of 336.47 g/mol). Their
    # occupancy figure is computed from an internal brain/effect-site
    # concentration produced by a multi-compartment brain PK submodel this
    # codebase does not have, so an exact percentage match isn't expected;
    # what we check is the same qualitative/quantitative regime, occupancy
    # clearly above 90% (near-saturating) at this clinically-cited overdose
    # concentration, consistent with simple two-state receptor theory using
    # the real Ki (a fentanyl concentration ~100x the 1.35-1.55 nM Ki range
    # from Volpe et al. 2011 must saturate the receptor almost completely
    # regardless of the exact brain/plasma partitioning).
    fentanyl, _naloxone = scenarios.mu_opioid_ligands_calibrated()
    site = ReceptorSite(name="site", ligands=[fentanyl])
    occ = np.array([0.0])
    C_overdose_nM = 148.6  # 50 ng/mL fentanyl, Moss et al. 2020
    dt = 0.01
    for _ in range(200000):
        occ = occ + dt * site.occupancy_rhs(occ, [C_overdose_nM])
    assert 0.90 < occ[0] <= 1.0


def build_calibrated_sim():
    return Simulation(
        pk_models=scenarios.mu_opioid_pk_models_calibrated(),
        receptor_site=scenarios.mu_opioid_receptor_site_calibrated(),
        network=scenarios.reward_circuit_network(),
        coupling=scenarios.mu_opioid_coupling(),
    )


def test_calibrated_agonist_raises_incentive_salience_above_baseline():
    # With real receptor/PK numbers substituted in place of the illustrative
    # placeholders, the qualitative v0 mechanism (mu-agonist disinhibits
    # VTA dopamine, raising the behavior readout above baseline) must still
    # hold; the circuit wiring itself (still illustrative) is unchanged.
    sim = build_calibrated_sim()
    readout = scenarios.incentive_salience_readout()
    t_eval = np.linspace(0, 60, 200)

    baseline = sim.run(dose_concentrations=[0.0, 0.0], t_span=(0, 60), t_eval=t_eval)
    agonist = sim.run(dose_concentrations=[148.6, 0.0], t_span=(0, 60), t_eval=t_eval)

    baseline_scores = np.array(
        [readout.score(sim.network, baseline["rates"][:, i]) for i in range(len(t_eval))]
    )
    agonist_scores = np.array(
        [readout.score(sim.network, agonist["rates"][:, i]) for i in range(len(t_eval))]
    )
    assert agonist_scores.max() > baseline_scores.max()


def test_naloxone_reverses_calibrated_agonist_effect():
    # Displacement should still work under real kinetics, provided naloxone
    # is given in large molar excess over the fentanyl already present, the
    # actual clinical mechanism (naloxone doses are typically a large molar
    # excess relative to fentanyl overdose concentrations), not naloxone
    # having a slower off-rate or higher affinity (it has neither; see
    # test_calibrated_fentanyl_koff_slower_than_naloxone_koff above).
    from scipy.integrate import solve_ivp

    sim = build_calibrated_sim()
    readout = scenarios.incentive_salience_readout()

    t_split = 30.0
    t_eval_1 = np.linspace(0, t_split, 150)
    phase1 = sim.run(
        dose_concentrations=[148.6, 0.0], t_span=(0, t_split), t_eval=t_eval_1
    )
    C0 = phase1["concentrations"][:, -1].copy()
    C0[1] += 3000.0  # naloxone bolus, ~20x fentanyl's molar concentration
    occ0 = phase1["occupancy"][:, -1]
    rates0 = phase1["rates"][:, -1]
    y0 = np.concatenate([C0, occ0, rates0])

    t_eval_2 = np.linspace(t_split, 60, 150)
    sol = solve_ivp(
        sim.rhs, (t_split, 60), y0, method="RK45", t_eval=t_eval_2, max_step=0.1
    )
    d, r = sim.d, sim.r
    occ_final = sol.y[d : d + r, -1]
    rates_final = sol.y[d + r : d + r + sim.c, -1]

    baseline = sim.run(
        dose_concentrations=[0.0, 0.0], t_span=(0, 60), t_eval=np.linspace(0, 60, 200)
    )
    baseline_final_score = readout.score(sim.network, baseline["rates"][:, -1])
    reversed_final_score = readout.score(sim.network, rates_final)

    assert occ_final[1] > occ_final[0]  # naloxone occupies more than fentanyl
    assert abs(reversed_final_score - baseline_final_score) < 1e-2
