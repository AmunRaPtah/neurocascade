"""P14.1: three-compartment dosed PK."""

import numpy as np
from scipy.integrate import solve_ivp

from neurocascade.pk import DosedThreeCompartmentPK


def test_iv_three_compartment_has_three_concentration_states_and_direct_q():
    model = DosedThreeCompartmentPK(
        route="IV", V1_L_per_kg=0.1, V2_L_per_kg=0.4, V3_L_per_kg=1.0,
        Q12_mL_per_min_per_kg=10.0, Q13_mL_per_min_per_kg=5.0,
        CL_mL_per_min_per_kg=20.0, body_weight_kg=70.0,
    )
    assert model.n_states == 3
    assert model.Q12_mL_per_min_per_kg == 10.0
    assert model.Q13_mL_per_min_per_kg == 5.0
    assert model.initial_state(1.0)[0] > 0.0
    assert model.initial_state(1.0)[1:].tolist() == [0.0, 0.0]


def test_extravascular_three_compartment_keeps_depot_in_coupled_state():
    model = DosedThreeCompartmentPK(
        route="PO", V1_L_per_kg=0.1, V2_L_per_kg=0.4, V3_L_per_kg=1.0,
        Q12_mL_per_min_per_kg=10.0, Q13_mL_per_min_per_kg=5.0,
        CL_mL_per_min_per_kg=20.0, body_weight_kg=70.0, F=0.8, ka_per_min=0.3,
    )
    state = model.initial_state(1.0)
    assert model.n_states == 4
    assert state[0] == 0.0 and state[3] > 0.0
    assert model.concentration_rhs(state).shape == (4,)


def test_three_compartment_mass_balance_eliminates_the_bioavailable_dose():
    model = DosedThreeCompartmentPK(
        route="IV", V1_L_per_kg=0.1, V2_L_per_kg=0.4, V3_L_per_kg=1.0,
        Q12_mL_per_min_per_kg=10.0, Q13_mL_per_min_per_kg=5.0,
        CL_mL_per_min_per_kg=20.0, body_weight_kg=1.0,
    )
    t = np.linspace(0, 5000, 20001)
    sol = solve_ivp(lambda _, y: model.concentration_rhs(y), (0, t[-1]), model.initial_state(1.0), t_eval=t, rtol=1e-10, atol=1e-13)
    assert sol.success
    elimination = model.CL_L_per_min * np.trapezoid(sol.y[0], t)
    assert elimination / model.dose_mg(1.0) > 0.999
