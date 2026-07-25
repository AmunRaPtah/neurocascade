import numpy as np

from neurocascade import Ligand, ReceptorSite


def test_single_ligand_occupancy_rises_with_concentration():
    agonist = Ligand(name="agonist", kon=1.0, koff=0.1, efficacy=1.0)
    site = ReceptorSite(name="site", ligands=[agonist])
    occ = np.array([0.0])
    d_occ = site.occupancy_rhs(occ, concentrations=[1.0])
    assert d_occ[0] > 0  # concentration present, no occupancy yet -> should rise


def test_occupancy_saturates_and_stays_within_bounds():
    agonist = Ligand(name="agonist", kon=1.0, koff=0.1, efficacy=1.0)
    site = ReceptorSite(name="site", ligands=[agonist])
    occ = np.array([0.0])
    concentration = [5.0]
    dt = 0.01
    for _ in range(200000):
        d_occ = site.occupancy_rhs(occ, concentration)
        occ = occ + dt * d_occ
    assert 0.0 <= occ[0] <= 1.0
    # steady state: kon*C*(1-occ) == koff*occ
    kon, koff = agonist.kon, agonist.koff
    expected = kon * concentration[0] / (kon * concentration[0] + koff)
    assert abs(occ[0] - expected) < 1e-3


def test_antagonist_has_zero_efficacy_activation():
    antagonist = Ligand(name="antagonist", kon=1.0, koff=0.1, efficacy=0.0)
    site = ReceptorSite(name="site", ligands=[antagonist])
    occ = np.array([0.8])
    assert site.activation(occ) == 0.0


def test_competitive_binding_occupancies_share_one_pool():
    agonist = Ligand(name="agonist", kon=1.0, koff=0.1, efficacy=1.0)
    antagonist = Ligand(name="antagonist", kon=1.0, koff=0.1, efficacy=0.0)
    site = ReceptorSite(name="site", ligands=[agonist, antagonist])
    occ = np.array([0.6, 0.3])
    concentrations = [1.0, 1.0]
    d_occ = site.occupancy_rhs(occ, concentrations)
    # free fraction is 1 - 0.6 - 0.3 = 0.1, shared by both: neither ligand
    # should behave as if it alone owned the full free pool.
    free_fraction = 1.0 - occ.sum()
    assert abs(free_fraction - 0.1) < 1e-9
    expected_agonist = agonist.kon * concentrations[0] * free_fraction - agonist.koff * occ[0]
    assert abs(d_occ[0] - expected_agonist) < 1e-9


def test_higher_affinity_antagonist_can_displace_bound_agonist():
    # High-affinity antagonist (very slow koff, fast kon) added at high
    # concentration should, over time, take over most of the receptor pool
    # even starting from an agonist-dominated state. This is the mechanism
    # naloxone displacement relies on.
    agonist = Ligand(name="agonist", kon=1.0, koff=0.15, efficacy=1.0)
    antagonist = Ligand(name="antagonist", kon=1.2, koff=0.05, efficacy=0.0)
    site = ReceptorSite(name="site", ligands=[agonist, antagonist])
    occ = np.array([0.8, 0.0])
    concentrations = [0.05, 3.0]  # agonist mostly cleared, antagonist bolus
    dt = 0.01
    for _ in range(200000):
        d_occ = site.occupancy_rhs(occ, concentrations)
        occ = occ + dt * d_occ
        occ = np.clip(occ, 0.0, 1.0)
    assert occ[1] > occ[0]
