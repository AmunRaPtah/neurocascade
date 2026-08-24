"""Real published validation targets, reused (not reinvented) from the
citations already inline in `scenarios.py`'s three "Calibrated: ..."
sections. This module is the single source of truth for "what real data do
we have to check a receptor-binding parameter against", distinct from
`priors.py` (what we believe before seeing that data) and `forward.py` (how
a parameter value maps to a comparable model prediction).

Every target below is an EQUILIBRIUM, SINGLE-LIGAND occupancy measurement
(fraction of receptor bound at one fixed drug concentration, no competing
ligand present), because that's the only kind of quantitative validation
target any of the three calibrated sections in `scenarios.py` actually
cites. No behavioral-readout (circuit-layer) target exists anywhere in this
codebase; see `docs/architecture.md`'s repeated "no literature target for
circuit weights/gains/taus" notes, so none is invented here. (Between P2.2
and P11.3 a rat-derived behavioural target lived here too; the species
pivot retired it to `legacy/behavioral_evidence.py`, and this module is
human/receptor-layer only again.)

Likelihood standard deviations: NONE of the three source papers reports a
statistical uncertainty on the occupancy PERCENTAGE itself (unlike the
kon/koff/Ki parameters in `priors.py`, several of which do have a real
reported sd). What every one of these three papers' figure DOES have,
documented in the existing calibrated tests
(`tests/test_scenarios_calibrated.py`, `tests/test_d2_scenario_calibrated.py`,
`tests/test_gaba_a_scenario_calibrated.py`), is a known, explicit, structural
reason this codebase's simplified one-compartment/no-protein-binding model
will not reproduce the reported percentage exactly (Moss et al.'s internal
brain-PK submodel this codebase lacks; haloperidol's ~89-93% plasma protein
binding; no dose-to-concentration PK submodel for the Savic et al. flumazenil
doses). Rather than pretend a specific sd was reported when it wasn't, every
target below uses ONE explicit, uniformly-applied, clearly-flagged modeling
allowance, `OCCUPANCY_LIKELIHOOD_SD = 0.07` (7 percentage points), chosen to
be generous enough to admit the known systematic gaps above while still
letting the likelihood meaningfully constrain the fitted parameter, in the
same spirit as the existing tests' own loose directional tolerances (e.g.
"occupancy > 0.90", "within 1e-2"), just expressed as a Gaussian sd instead
of a pass/fail bound so it can be used as a PyMC likelihood. This is an
engineering judgment call, not a literature-reported figure, and is reported
as such rather than disguised as real measurement uncertainty.
"""

from .forward import ng_per_ml_to_nM

OCCUPANCY_LIKELIHOOD_SD = 0.07

# ---------------------------------------------------------------------------
# 1. Mu-opioid / fentanyl: Moss et al. 2020 (PLoS ONE,
#    doi:10.1371/journal.pone.0234683), same fentanyl Ki (~1.35 nM) as this
#    codebase's Volpe et al. 2011-derived value. Reports baseline (no
#    naloxone) mu-receptor occupancy of 73%/97%/98.9% at fentanyl exposures
#    of 25/50/75 ng/mL; the existing test
#    (test_calibrated_fentanyl_occupancy_near_saturating_at_overdose_
#    concentration) uses the 50 ng/mL/97% point, reused here unchanged.
# ---------------------------------------------------------------------------
FENTANYL_MW = 336.47  # g/mol
MU_OPIOID_FENTANYL_OCCUPANCY_EVIDENCE = {
    "citation": "Moss et al. 2020, PLoS ONE, doi:10.1371/journal.pone.0234683",
    "concentration_nM": 148.6,  # 50 ng/mL fentanyl
    "occupancy_target_mean": 0.97,
    "occupancy_target_sd": OCCUPANCY_LIKELIHOOD_SD,
}

# ---------------------------------------------------------------------------
# 2. D2 / haloperidol: Kapur et al. 1997 (Psychopharmacology,
#    doi:10.1007/s002130050277), PET occupancy vs. plasma level, already used
#    in test_d2_scenario_calibrated.py's
#    test_calibrated_haloperidol_occupancy_near_pet_occupancy_targets. Two
#    points on the same saturating-hyperbola fit: 50% occupancy at 0.51
#    ng/mL, 80% occupancy at 2.0 ng/mL.
# ---------------------------------------------------------------------------
HALOPERIDOL_MW = 375.86  # g/mol
D2_HALOPERIDOL_OCCUPANCY_EVIDENCE = {
    "citation": "Kapur et al. 1997, Psychopharmacology, doi:10.1007/s002130050277",
    "ed50_concentration_nM": ng_per_ml_to_nM(0.51, HALOPERIDOL_MW),
    "ed50_occupancy_target_mean": 0.50,
    "ed80_concentration_nM": ng_per_ml_to_nM(2.0, HALOPERIDOL_MW),
    "ed80_occupancy_target_mean": 0.80,
    "occupancy_target_sd": OCCUPANCY_LIKELIHOOD_SD,
}

# ---------------------------------------------------------------------------
# 3. GABA-A / flumazenil: Savic, Widen, Stone-Elander 1991 (Lancet,
#    doi:10.1016/0140-6736(91)90799-u), already used in
#    test_gaba_a_scenario_calibrated.py's
#    test_calibrated_flumazenil_occupancy_shape_matches_savic_1991. A 1.5 mg
#    IV dose occupied ~55% of human benzodiazepine binding sites; a 10x
#    higher dose (15 mg) occupied "nearly all" of them (no exact percentage
#    reported for the high dose in the paper). No dose->concentration PK
#    submodel exists in this codebase (no volume of distribution/absorption
#    kinetics), so, exactly as the existing test does, the low-dose point is
#    converted into an equilibrium concentration via the closed-form
#    two-state relationship at the reference Ki (1.0 nM, Mohler et al. 1981)
#    : C_low = Ki * 0.55 / (1 - 0.55), and the high-dose point uses 10x
#    that concentration. This reuses the SAME fixed concentration constants
#    the existing test already derived; what's new here is treating the two
#    reported occupancy figures (55% and, per the "nearly all" language, an
#    assigned representative 95%) as Gaussian likelihood targets on
#    flumazenil's Ki, rather than as a pass/fail assertion. The high-dose
#    95% figure is explicitly a stand-in for a qualitative "nearly all"
#    claim, not a number Savic et al. reported directly; flagged as such.
# ---------------------------------------------------------------------------
_REFERENCE_KI_FLUMAZENIL_NM = 1.0  # Mohler et al. 1981 point estimate
_C_LOW_NM = _REFERENCE_KI_FLUMAZENIL_NM * 0.55 / (1.0 - 0.55)
GABA_A_FLUMAZENIL_OCCUPANCY_EVIDENCE = {
    "citation": "Savic, Widen, Stone-Elander 1991, Lancet, doi:10.1016/0140-6736(91)90799-u",
    "low_dose_concentration_nM": _C_LOW_NM,
    "low_dose_occupancy_target_mean": 0.55,
    "high_dose_concentration_nM": 10.0 * _C_LOW_NM,
    "high_dose_occupancy_target_mean": 0.95,  # stand-in for "nearly all"; see note above
    "high_dose_is_qualitative_stand_in": True,
    "occupancy_target_sd": OCCUPANCY_LIKELIHOOD_SD,
}
