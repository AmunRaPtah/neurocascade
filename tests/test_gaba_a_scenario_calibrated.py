"""Tests for the calibrated GABA-A/benzodiazepine-site parameter set
(scenarios.py, "Calibrated GABA-A / benzodiazepine-site receptor +
diazepam/flumazenil PK" section). See that section's comments for full
source citations; each test below cites the specific literature figure it
checks against. Mirrors the structure of test_scenarios_calibrated.py.

P11.3 split this suite in two: the receptor and PK constants stay CALIBRATED
(Ki values are receptor-level measurements, not behavioural fits), while the
connection-modulation gain P2.2 fitted against rat behaviour was retired
with the rodent bridge. What remains here pins that split - the calibrated
stack now composes the ILLUSTRATIVE modulation, no `*_calibrated` accessor
for it exists any more, and the retired posterior survives only as the
historical artifact under results/calibration/.
"""

import csv
import os

import numpy as np
from scipy.integrate import solve_ivp

from neurocascade import scenarios
from neurocascade.engine import Simulation
from neurocascade.receptor import ReceptorSite


def _run_to_equilibrium(ligand, Ki, C, steps=200000, dt=0.01):
    site = ReceptorSite(name="site", ligands=[ligand])
    occ = np.array([0.0])
    for _ in range(steps):
        occ = occ + dt * site.occupancy_rhs(occ, [C])
    return occ[0]


def test_calibrated_kon_koff_reproduce_ki_at_equilibrium():
    # Sanity check on the koff = kon_shared * Ki derivation itself (see
    # scenarios.py): for a single ligand at equilibrium,
    # occ = kon*C / (kon*C + koff) = C / (C + Ki). At C == Ki that must
    # equal 0.5 for both ligands, or the derivation has a units/arithmetic
    # error.
    diazepam, flumazenil = scenarios.gaba_a_benzodiazepine_ligands_calibrated()
    Ki_diazepam, Ki_flumazenil = 3.6, 1.0  # nM; Mohler & Okada 1977, Mohler et al. 1981

    for ligand, Ki in [(diazepam, Ki_diazepam), (flumazenil, Ki_flumazenil)]:
        occ = _run_to_equilibrium(ligand, Ki, C=Ki)
        assert abs(occ - 0.5) < 1e-3


def test_calibrated_flumazenil_binds_more_tightly_than_diazepam():
    # Mohler et al. 1981 (J Neurochem): [3H]Ro 15-1788 (flumazenil) Kd = 1.0
    # nM in rat cerebral cortex, versus Mohler & Okada 1977's [3H]diazepam
    # Kd = 3.6 nM in the same preparation, i.e. flumazenil binds this site
    # MORE TIGHTLY than diazepam. Under this codebase's shared-kon
    # assumption (see scenarios.py derivation note), a tighter Ki means a
    # SLOWER koff (longer receptor residence time). This is the opposite of
    # the illustrative placeholder scenario, which assumed flumazenil had
    # the faster koff (shorter residence, "weaker hold") of the two, to
    # mirror flumazenil's short CLINICAL duration of action. Real receptor-
    # level affinity data says flumazenil's short clinical duration comes
    # from its fast PLASMA clearance and the absence of a lingering active
    # metabolite (unlike diazepam's desmethyldiazepam), not from weaker or
    # faster receptor binding; see gaba_a_pk_models_calibrated() for the PK
    # side of this same point.
    diazepam, flumazenil = scenarios.gaba_a_benzodiazepine_ligands_calibrated()
    assert flumazenil.koff < diazepam.koff


def test_calibrated_flumazenil_occupancy_shape_matches_savic_1991():
    # Validation target: Savic I, Widen L, Stone-Elander S. Lancet.
    # 1991;337(8734):133-137 report that a 1.5 mg IV dose of flumazenil in
    # humans occupied ~55% of benzodiazepine binding sites, while a 10x
    # higher dose (15 mg) occupied nearly all of them. This codebase has no
    # dose->plasma/brain-concentration model (no volume of distribution or
    # absorption kinetics), so we cannot reproduce the exact 1.5 mg/15 mg
    # dose mapping; what we validate is the SHAPE of that relationship
    # using the calibrated Ki alone: a concentration chosen to give ~55%
    # occupancy at equilibrium (simple two-state receptor theory,
    # occ = C / (C + Ki)) must, at 10x that concentration, land close to
    # saturation, exactly the qualitative jump Savic et al. observed
    # (55% -> "nearly all") going from 1.5 mg to 15 mg.
    diazepam, flumazenil = scenarios.gaba_a_benzodiazepine_ligands_calibrated()
    Ki_flumazenil = 1.0  # nM, Mohler et al. 1981
    C_low = Ki_flumazenil * 0.55 / 0.45  # solve occ=C/(C+Ki) for occ=0.55

    occ_low = _run_to_equilibrium(flumazenil, Ki_flumazenil, C=C_low)
    occ_high = _run_to_equilibrium(flumazenil, Ki_flumazenil, C=10 * C_low)

    assert abs(occ_low - 0.55) < 1e-2
    assert occ_high > 0.90


def test_calibrated_pk_clearance_rates_are_similar_at_acute_timescale():
    # Documents, and guards against silently losing, the genuinely
    # interesting finding recorded in gaba_a_pk_models_calibrated(): at this
    # scenario's acute (minutes-to-~1h) timescale, diazepam's real
    # distribution-phase clearance (Mandelli et al. 1978, ~60 min half-
    # life) and flumazenil's real elimination clearance (FDA Romazicon
    # label, ~54 min half-life) are close in magnitude, NOT the dramatic
    # long-acting-vs-short-acting asymmetry (ke ratio 25x) the illustrative
    # placeholder assumed. The real long-vs-short asymmetry between these
    # two drugs only appears on the multi-hour-to-multi-day scale, driven
    # by diazepam's active metabolite, which this one-compartment model
    # does not represent.
    ke_diazepam, ke_flumazenil = (pk.ke for pk in scenarios.gaba_a_pk_models_calibrated())
    ratio = max(ke_diazepam, ke_flumazenil) / min(ke_diazepam, ke_flumazenil)
    assert ratio < 2.0


def build_calibrated_sim():
    # The calibrated receptor + PK stack composed with the ILLUSTRATIVE
    # modulation - the only composition available since P11.3 retired the
    # fitted gain, and exactly what
    # `scenarios.gaba_a_receptor_system_calibrated()` documents.
    return Simulation(
        pk_models=scenarios.gaba_a_pk_models_calibrated(),
        receptor_site=scenarios.gaba_a_receptor_site_calibrated(),
        network=scenarios.amygdala_anxiety_circuit_network(),
        coupling=scenarios.gaba_a_coupling(),
        connection_modulation=scenarios.gaba_a_connection_modulation(),
    )


def test_the_retired_fit_is_gone_and_its_artifact_says_what_it_was():
    """P2.2's gain did not survive P11.3, and neither did its accessor.

    Three things pinned together. The committed posterior summary stays
    under `results/calibration/` UNCHANGED, as a historical record of a fit
    that no longer informs anything. Production carries the restored
    hand-set 0.5, which is NOT the fitted value - the retirement must be
    visible in the numbers, not only in prose. And there is deliberately no
    `gaba_a_connection_modulation_calibrated()` any more: a function by
    that name returning hand-set numbers would be the exact dishonesty
    scenarios.py's illustrative/calibrated split exists to prevent.
    """
    summary_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "calibration", "gaba_a_summary.csv"
    )
    with open(summary_path) as handle:
        rows = {row[""]: row for row in csv.DictReader(handle)}
    posterior_mean = float(rows["gaba_a_connection_modulation_gain"]["mean"])
    # The historical artifact still records the fit it was written from.
    assert abs(posterior_mean - 0.0742) <= 0.0005

    assert not hasattr(scenarios, "gaba_a_connection_modulation_calibrated")
    modulation = scenarios.gaba_a_connection_modulation()
    assert set(modulation) == {("BLA", "GABA_Interneuron")}
    (gain,) = modulation.values()
    assert gain == scenarios.GABA_A_CONNECTION_MODULATION_GAIN == 0.5
    # ...and production is demonstrably not carrying the old fit quietly.
    assert abs(gain - posterior_mean) > 0.01


def test_calibrated_diazepam_lowers_anxiety_index_below_baseline():
    # With real receptor/PK numbers substituted in place of the
    # illustrative placeholders, the qualitative v0 mechanism (diazepam
    # potentiates the GABAergic interneuron -> BLA synapse via
    # connection_modulation, suppressing BLA and, downstream, anxiety_index)
    # must still hold; the circuit wiring and potentiation gain are the
    # ILLUSTRATIVE ones - see gaba_a_connection_modulation() and the P11.3
    # record on GABA_A_CONNECTION_MODULATION_GAIN in scenarios.py.
    sim = build_calibrated_sim()
    readout = scenarios.anxiety_index_readout()
    t_eval = np.linspace(0, 40, 200)

    baseline = sim.run(dose_concentrations=[0.0, 0.0], t_span=(0, 40), t_eval=t_eval)
    diazepam = sim.run(dose_concentrations=[100.0, 0.0], t_span=(0, 40), t_eval=t_eval)

    baseline_final = readout.score(sim.network, baseline["rates"][:, -1])
    diazepam_final = readout.score(sim.network, diazepam["rates"][:, -1])
    assert diazepam_final < baseline_final - 0.1


def test_calibrated_flumazenil_alone_stays_near_baseline():
    # Flumazenil has efficacy=0 (no potentiation of its own) and there is no
    # diazepam present to displace, so it should do ~nothing on its own,
    # exactly like the illustrative scenario's equivalent test.
    sim = build_calibrated_sim()
    readout = scenarios.anxiety_index_readout()
    t_eval = np.linspace(0, 40, 200)

    baseline = sim.run(dose_concentrations=[0.0, 0.0], t_span=(0, 40), t_eval=t_eval)
    flumazenil = sim.run(dose_concentrations=[0.0, 100.0], t_span=(0, 40), t_eval=t_eval)

    baseline_final = readout.score(sim.network, baseline["rates"][:, -1])
    flumazenil_final = readout.score(sim.network, flumazenil["rates"][:, -1])
    assert abs(flumazenil_final - baseline_final) < 1e-2


def test_calibrated_flumazenil_partially_reverses_diazepam():
    """Two-phase dosing (diazepam first, flumazenil injected mid-course),
    the same reversal pattern as test_gaba_a_scenario.py and the mu-opioid
    calibrated suite's naloxone-reversal test. Displacement at the shared
    benzodiazepine-site pool (ordinary ReceptorSite competitive occupancy)
    should still work under real Ki-derived kinetics, provided flumazenil is
    given in molar excess over the diazepam already present, the same
    "displace by dosing, not by tighter binding or a slower off-rate" caveat
    made for naloxone/fentanyl (flumazenil already has, if anything, the
    SLOWER off-rate here; see
    test_calibrated_flumazenil_binds_more_tightly_than_diazepam above)."""
    sim = build_calibrated_sim()
    readout = scenarios.anxiety_index_readout()
    t_span = (0, 40)
    t_split = 20.0

    t_eval_1 = np.linspace(t_span[0], t_split, 200)
    phase1 = sim.run(
        dose_concentrations=[100.0, 0.0], t_span=(t_span[0], t_split), t_eval=t_eval_1
    )
    score_at_split = readout.score(sim.network, phase1["rates"][:, -1])

    C0 = phase1["concentrations"][:, -1].copy()
    C0[1] += 2000.0  # flumazenil bolus injected at t_split, large molar excess
    occ0 = phase1["occupancy"][:, -1]
    rates0 = phase1["rates"][:, -1]
    y0 = np.concatenate([C0, occ0, rates0])

    t_eval_2 = np.linspace(t_split, t_span[1], 200)
    sol = solve_ivp(
        sim.rhs, (t_split, t_span[1]), y0, method="RK45", t_eval=t_eval_2, max_step=0.1
    )
    d, r = sim.d, sim.r
    occ_final = sol.y[d : d + r, -1]
    rates2 = sol.y[d + r : d + r + sim.c]
    scores_after = np.array(
        [readout.score(sim.network, rates2[:, i]) for i in range(len(t_eval_2))]
    )

    assert occ_final[1] > occ_final[0]  # flumazenil occupies more than diazepam
    assert scores_after.max() > score_at_split  # pushed back up toward baseline


def test_state_arrays_have_expected_shapes():
    sim = build_calibrated_sim()
    t_eval = np.linspace(0, 10, 50)
    result = sim.run(dose_concentrations=[1.0, 0.0], t_span=(0, 10), t_eval=t_eval)
    assert result["concentrations"].shape == (2, 50)
    assert result["occupancy"].shape == (2, 50)
    assert result["rates"].shape == (3, 50)  # GABA_Interneuron, BLA, CeA
    assert np.all(result["occupancy"] >= 0.0)
    assert np.all(result["occupancy"] <= 1.0)
