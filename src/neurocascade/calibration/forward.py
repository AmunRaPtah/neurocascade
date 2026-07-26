"""Forward-model helpers for Bayesian calibration.

All five validation targets used by `bayesian_calibration.py` (see
`evidence.py`) are equilibrium, single-ligand receptor-occupancy
measurements: "what fraction of receptor is occupied once a fixed plasma
concentration has had time to equilibrate", with no second competing ligand
present. This module computes that quantity by integrating the actual
`ReceptorSite.occupancy_rhs` ODE via `scipy.integrate.solve_ivp` (the same
integrator `engine.py` uses for the full coupled simulation) out to a time
long enough to reach steady state, rather than substituting the closed-form
two-state formula `occ = C / (C + Kd)` directly. The closed form is
algebraically equivalent and is exactly what the existing calibrated tests
in `tests/test_scenarios_calibrated.py` etc. already check numerically (via
a manual Euler loop); integrating the real ODE here keeps this calibration
scaffold honestly built on `ReceptorSite`, the one place that mechanism is
actually defined, per this track's brief to reuse existing classes rather
than reimplement simulation logic.

A genuine identifiability point, reported here rather than glossed over: an
equilibrium, single-ligand occupancy curve depends only on the ratio
Kd = koff / kon, not on kon and koff individually (the ODE's approach to
that same equilibrium is what would depend on them separately, but none of
the three validation targets probes approach dynamics). This matters for two
of this module's three receptor systems:

  - mu-opioid (fentanyl) and GABA-A (flumazenil) each have only a Ki *point
    estimate* with no reported kon/koff split, so `equilibrium_occupancy_
    from_kd` below fixes an arbitrary reference kon (kon=1.0 nM^-1 min^-1)
    and sets koff = Kd accordingly, purely so the real ODE is what gets
    integrated. The absolute value of that reference kon is immaterial to
    the equilibrium result; only Kd is fit.
  - D2 (haloperidol) has independently reported kon AND koff (Sykes et al.
    2017), each with its own real standard deviation, so both are sampled
    as separate free parameters with their own informed priors; the Kapur
    et al. 1997 occupancy likelihood constrains their *ratio*
    (Kd = koff/kon), while each parameter's own prior regularizes it
    individually. `equilibrium_occupancy` below is called directly with
    both sampled values for that system.
"""

import numpy as np
from scipy.integrate import solve_ivp

from ..receptor import Ligand, ReceptorSite

# Reference association rate constant used only when a system's calibrated
# literature source reports Ki/Kd but not an independent kon/koff split
# (mu-opioid fentanyl, GABA-A flumazenil/diazepam). See module docstring:
# only the Kd = koff/kon ratio affects an equilibrium single-ligand
# occupancy curve, so this value's magnitude is arbitrary and does not
# affect any fitted quantity, provided it is nonzero and finite.
_REFERENCE_KON = 1.0  # nM^-1 min^-1


def equilibrium_occupancy(kon, koff, concentration, efficacy=1.0):
    """Integrate `ReceptorSite.occupancy_rhs` for one ligand at a fixed
    plasma concentration out to steady state and return final fractional
    occupancy. `efficacy` doesn't affect occupancy (only `activation`), it's
    accepted for API symmetry/readability at call sites.
    """
    if concentration <= 0.0:
        return 0.0
    ligand = Ligand(name="probe", kon=kon, koff=koff, efficacy=efficacy)
    site = ReceptorSite(name="single_ligand_equilibrium_probe", ligands=[ligand])

    # The occupancy ODE here (a single competing ligand) is linear:
    # d(occ)/dt = kon*C - (kon*C + koff)*occ, i.e. a first-order relaxation
    # toward equilibrium with rate constant k = kon*C + koff. Integrating to
    # t_max = 30/k guarantees occupancy within exp(-30) ~ 1e-13 of
    # equilibrium regardless of the specific kon/koff/concentration values
    # used across this module's three receptor systems.
    k = kon * concentration + koff
    k = max(k, 1e-9)
    t_max = 30.0 / k

    def rhs(t, occ):
        return site.occupancy_rhs(occ, [concentration])

    sol = solve_ivp(rhs, (0.0, t_max), [0.0], method="RK45", rtol=1e-9, atol=1e-12)
    if not sol.success:
        raise RuntimeError("equilibrium occupancy integration failed: %s" % sol.message)
    return float(sol.y[0, -1])


def equilibrium_occupancy_from_kd(kd, concentration, efficacy=1.0):
    """Same as `equilibrium_occupancy`, but parametrized by Kd directly
    (fixing an arbitrary reference kon; see module docstring) for the two
    ligands (fentanyl, flumazenil) whose literature source reports only a
    Ki/Kd point estimate, not an independently measured kon/koff.
    """
    koff = _REFERENCE_KON * kd
    return equilibrium_occupancy(_REFERENCE_KON, koff, concentration, efficacy=efficacy)


def ng_per_ml_to_nM(ng_per_ml, molar_mass_g_per_mol):
    """Unit conversion shared by the D2/haloperidol validation target (Kapur
    et al. 1997 report plasma levels in ng/mL); reproduces the conversion
    already used ad hoc in `tests/test_d2_scenario_calibrated.py`, pulled
    out here so `evidence.py` has one source of truth for it.
    """
    return (ng_per_ml * 1e-6 / molar_mass_g_per_mol) * 1e9
