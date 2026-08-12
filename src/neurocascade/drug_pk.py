"""Literature PK parameter sets, one drug / species / route at a time.

`pk.py` deliberately contains no drug-specific constant of any kind: it is
the structure (routes, compartments, absorption, the dose -> state
arithmetic), and nothing else. This module is where measured numbers for
actual drugs live, each one carrying its own citation, species, route and
unit conversion, in the same style as the calibrated blocks in
`scenarios.py`.

Rules this module follows, and the reason each one exists
--------------------------------------------------------

1. Every number is either quoted from a named source, or derived in code
   from numbers that are (with the derivation written out), or absent. A
   parameter that could not be sourced is *not* filled in with a plausible
   value: the constructor for that route raises `UnsourcedParameterError`
   and the search that came up empty is recorded in
   `SEARCHED_AND_EMPTY` below and in docs/architecture.md.
2. No cross-species and no cross-drug substitution. Rat haloperidol
   parameters come from rat haloperidol measurements or they do not exist
   here.
3. Body weight is never defaulted. Dose, volumes and clearances are all per
   kg, so weight cancels out of the concentration-time course, but it does
   not cancel out of the absolute quantities (`dose_mg`, `CL_L_per_min`)
   that published tables report, and picking "a typical rat" silently is
   exactly the kind of invisible assumption this codebase is trying not to
   make.

Currently covered
-----------------

    haloperidol, rat, IV   -- Cheng & Paalzow 1992 (two-compartment, and a
                              documented one-compartment collapse of it)
    haloperidol, rat, IP   -- NOT AVAILABLE, see SEARCHED_AND_EMPTY
    haloperidol, rat, SC   -- NOT AVAILABLE, see SEARCHED_AND_EMPTY
    fentanyl,    rat, IV   -- Yassen et al. 2008 (two-compartment population
                              model, and a documented one-compartment
                              collapse of it)
    fentanyl,    rat, SC   -- disposition parameters sourced
                              (Canfield & Sprague 2025) but ka absent, so no
                              model is built; see SEARCHED_AND_EMPTY
    fentanyl,    rat, IP   -- disposition parameters sourced (Almousa et al.
                              2011) but mutually inconsistent and no F; see
                              SEARCHED_AND_EMPTY
    fentanyl,    rat, PO   -- F sourced (Schneider & Brune 1985, 1.5%) but
                              ka absent; see SEARCHED_AND_EMPTY
    fentanyl,    mouse, *  -- NOT AVAILABLE for any route, see
                              SEARCHED_AND_EMPTY

Diazepam (P1.4) is not here yet.
"""

import math

from .pk import (
    LN2,
    DosedOneCompartmentPK,
    DosedTwoCompartmentPK,
    intercompartmental_clearance_from_terminal_rate,
)


class UnsourcedParameterError(LookupError):
    """Raised when a drug/species/route combination is requested whose
    required parameters have been searched for and not found.

    A distinct exception type rather than a generic error because "this
    number does not exist in the literature we could reach" is a *result*
    in this codebase, not a bug: see `SEARCHED_AND_EMPTY`.
    """


# ---------------------------------------------------------------------------
# Haloperidol
# ---------------------------------------------------------------------------
#
# Sources
# -------
#   [ChengPaalzow1992] Cheng YF, Paalzow LK. "Linear pharmacokinetics of
#                haloperidol in the rat." Biopharm Drug Dispos.
#                1992;13(1):69-76. doi:10.1002/bdd.2510130106, PMID 1554878.
#                Rat, intravenous bolus at 0.5, 1.0 and 2.5 mg/kg, plus
#                12 h constant infusions. Abstract quoted verbatim from
#                PubMed: "haloperidol was a high extraction ratio drug with
#                a total blood clearance averaging 83 ml min-1 kg-1. The
#                volumes of distribution were large with a mean of 5.5 l
#                kg-1 (Vc), 11.1 l kg-1 (V beta), and 9.6 l kg-1 (Vss),
#                respectively. The terminal half-life was 1.5 h. The
#                disposition kinetics of haloperidol was found to be linear
#                over the dose range studied." Disposition is explicitly
#                biexponential (the paper's own infusion predictions use a
#                biexponential model), which is why the primary parameter
#                set below is two-compartment.
#                Limits of what this source gives, checked and recorded
#                rather than guessed: the full text is closed access
#                (Unpaywall: is_oa false, no repository copy), so rat
#                strain, sex, body weight, anaesthesia state and assay were
#                not obtainable; and the reported quantities are BLOOD
#                clearance and blood-based volumes, so concentrations
#                produced by this parameter set are whole-blood
#                concentrations (see the blood:plasma note in
#                SEARCHED_AND_EMPTY).
#   [Kapur2003]  Kapur S, VanderSpek SC, Brownlee BA, Nobrega JN.
#                "Antipsychotic dosing in preclinical models is often
#                unrepresentative of the clinical condition: a suggested
#                solution based on in vivo occupancy." J Pharmacol Exp Ther.
#                2003;305(2):625-31. doi:10.1124/jpet.102.046987,
#                PMID 12606608. Not a source of any number used here; cited
#                as independent corroboration that rodent antipsychotic
#                elimination is fast ("the half-life of antipsychotics in
#                rodents is 4 to 6 times faster than in humans"), which is
#                the qualitative claim the 1.5 h rat terminal half-life
#                makes against haloperidol's ~14-37 h in humans
#                ([Kudo1999], cited in scenarios.py).
#   [DaviesMorris1993] Davies B, Morris T. "Physiological parameters in
#                laboratory animals and humans." Pharm Res.
#                1993;10(7):1093-5. doi:10.1023/A:1018943613122,
#                PMID 8378254. The standard physiological-parameter
#                compilation; rat hepatic blood flow 13.8 mL/min in a 250 g
#                rat, i.e. ~55 mL/min/kg. Used ONLY qualitatively below, to
#                state why an IP bioavailability for this drug cannot be
#                assumed to be near 1 - never to compute one.
#   [PubChem3559] PubChem CID 3559, haloperidol, C21H23ClFNO2, molecular
#                weight 375.9 g/mol. A chemical fact, used for the mg/L ->
#                nM conversion; the value 375.86 below is the same number
#                already used by `scenarios.d2_*_calibrated` for the
#                Kapur 1997 occupancy conversion, kept identical so the two
#                cannot drift apart.

HALOPERIDOL_MOLAR_MASS_G_PER_MOL = 375.86  # [PubChem3559]

#: Total *blood* clearance, mL/min/kg [ChengPaalzow1992].
HALOPERIDOL_RAT_IV_CL_ML_PER_MIN_PER_KG = 83.0
#: Central-compartment volume Vc, L/kg [ChengPaalzow1992].
HALOPERIDOL_RAT_IV_VC_L_PER_KG = 5.5
#: Steady-state volume of distribution Vss, L/kg [ChengPaalzow1992].
HALOPERIDOL_RAT_IV_VSS_L_PER_KG = 9.6
#: Terminal-phase volume of distribution V_beta = CL/beta, L/kg
#: [ChengPaalzow1992].
HALOPERIDOL_RAT_IV_VBETA_L_PER_KG = 11.1
#: Terminal half-life as reported, minutes: "1.5 h" [ChengPaalzow1992].
HALOPERIDOL_RAT_IV_TERMINAL_HALF_LIFE_MIN = 1.5 * 60.0
#: The interval that "1.5 h", reported to two significant figures, actually
#: constrains the terminal half-life to: 1.45-1.55 h in minutes. Used as the
#: tolerance in `tests/test_drug_pk.py` instead of an invented percentage,
#: because the honest question is not "how close is the model to 90.0 min"
#: but "is the model inside the interval the paper's own rounding leaves
#: open".
HALOPERIDOL_RAT_IV_TERMINAL_HALF_LIFE_INTERVAL_MIN = (1.45 * 60.0, 1.55 * 60.0)

# Peripheral volume, derived: Vss = V1 + V2 by definition of Vss for a
# two-compartment model, so V2 = Vss - Vc = 9.6 - 5.5 L/kg. Both inputs are
# [ChengPaalzow1992] values; no new information is introduced.
HALOPERIDOL_RAT_IV_V2_L_PER_KG = (
    HALOPERIDOL_RAT_IV_VSS_L_PER_KG - HALOPERIDOL_RAT_IV_VC_L_PER_KG
)

# Terminal disposition rate constant beta, derived. Two of the paper's own
# numbers determine it, and they are not perfectly consistent with each
# other at the precision they were printed:
#
#   from V_beta:      beta = CL / V_beta = 83 / 11100        = 0.0074775/min
#                            -> t_half = 92.7 min = 1.545 h
#   from t_half:      beta = ln2 / 90 min                    = 0.0077016/min
#                            -> V_beta = CL/beta = 10.8 L/kg
#
# We anchor on V_beta, for a stated reason rather than convenience: V_beta
# is reported to three significant figures and "1.5 h" to two, and the
# half-life the V_beta anchor implies (92.7 min, i.e. 1.545 h) lies INSIDE
# the interval that "1.5 h" rounds from (1.45-1.55 h), while the other
# anchor puts V_beta at 10.8 against a reported 11.1. Anchoring the other
# way is a one-line change and moves the derived Q by ~24% (see
# `HALOPERIDOL_RAT_IV_Q_SENSITIVITY_ML_PER_MIN_PER_KG`), which is itself
# worth knowing and is why both numbers are recorded here.
HALOPERIDOL_RAT_IV_BETA_PER_MIN = (
    HALOPERIDOL_RAT_IV_CL_ML_PER_MIN_PER_KG
    / (HALOPERIDOL_RAT_IV_VBETA_L_PER_KG * 1000.0)
)

# Inter-compartmental clearance Q, derived from (Vc, V2, CL, beta) by
# `pk.intercompartmental_clearance_from_terminal_rate`. [ChengPaalzow1992]
# does not report Q - almost no in vivo PK paper does - and a
# two-compartment model cannot be built without it, so it is solved for
# rather than chosen. It is a DERIVED number, not a measured one, and the
# derivation is ill-conditioned (see the module note in `pk.py`): the two
# self-consistent readings of the same paper give 114.5 mL/min/kg (V_beta
# anchor, used here) and 141.6 mL/min/kg (half-life anchor), which is a 24%
# spread on a parameter neither reading measured.
HALOPERIDOL_RAT_IV_Q_ML_PER_MIN_PER_KG = (
    intercompartmental_clearance_from_terminal_rate(
        V1_L_per_kg=HALOPERIDOL_RAT_IV_VC_L_PER_KG,
        V2_L_per_kg=HALOPERIDOL_RAT_IV_V2_L_PER_KG,
        CL_mL_per_min_per_kg=HALOPERIDOL_RAT_IV_CL_ML_PER_MIN_PER_KG,
        beta_per_min=HALOPERIDOL_RAT_IV_BETA_PER_MIN,
    )
)
#: (Q from the V_beta anchor, Q from the "1.5 h" anchor), mL/min/kg. The
#: spread between them is the real uncertainty on Q implied by the source's
#: own reporting precision, and any downstream use of the distribution phase
#: (P1.6's Cmax/Tmax validation especially) has to live with it.
HALOPERIDOL_RAT_IV_Q_SENSITIVITY_ML_PER_MIN_PER_KG = (
    HALOPERIDOL_RAT_IV_Q_ML_PER_MIN_PER_KG,
    intercompartmental_clearance_from_terminal_rate(
        V1_L_per_kg=HALOPERIDOL_RAT_IV_VC_L_PER_KG,
        V2_L_per_kg=HALOPERIDOL_RAT_IV_V2_L_PER_KG,
        CL_mL_per_min_per_kg=HALOPERIDOL_RAT_IV_CL_ML_PER_MIN_PER_KG,
        beta_per_min=LN2 / HALOPERIDOL_RAT_IV_TERMINAL_HALF_LIFE_MIN,
    ),
)

#: Bioavailability of the IV route: 1 by definition, not by measurement.
HALOPERIDOL_RAT_IV_F = 1.0


def haloperidol_rat_iv_pk(
    body_weight_kg, molar_mass_g_per_mol=HALOPERIDOL_MOLAR_MASS_G_PER_MOL
):
    """Two-compartment rat haloperidol PK after an IV bolus
    [ChengPaalzow1992]; dose in mg/kg.

    This is the primary parameter set for the drug: the source reports
    biexponential disposition, and a two-compartment model is the one that
    can carry both the fast distribution phase (Vc 5.5 L/kg, so an IV bolus
    starts at Dose/5.5 rather than Dose/11.1) and the 1.5 h terminal phase
    with their own literature values instead of collapsing them.

    Concentrations are whole blood, in nM by default (pass
    `molar_mass_g_per_mol=None` for mg/L).
    """
    return DosedTwoCompartmentPK(
        route="IV",
        V1_L_per_kg=HALOPERIDOL_RAT_IV_VC_L_PER_KG,
        V2_L_per_kg=HALOPERIDOL_RAT_IV_V2_L_PER_KG,
        Q_mL_per_min_per_kg=HALOPERIDOL_RAT_IV_Q_ML_PER_MIN_PER_KG,
        CL_mL_per_min_per_kg=HALOPERIDOL_RAT_IV_CL_ML_PER_MIN_PER_KG,
        body_weight_kg=body_weight_kg,
        molar_mass_g_per_mol=molar_mass_g_per_mol,
        name="haloperidol/rat/IV (Cheng & Paalzow 1992)",
    )


def haloperidol_rat_iv_one_compartment_pk(
    body_weight_kg, molar_mass_g_per_mol=HALOPERIDOL_MOLAR_MASS_G_PER_MOL
):
    """One-compartment collapse of the same source, for continuity with the
    existing single-exponential scenarios.

    `Vd = V_beta = 11.1 L/kg` rather than `Vc`, because the point of the
    collapse is to keep the terminal phase right: with CL = 83 mL/min/kg it
    reproduces the reported 1.5 h half-life (92.7 min) exactly by
    construction, and `AUC = Dose/CL` is unchanged, since AUC depends only
    on clearance. What it necessarily gets wrong is the first few minutes:
    it starts at `Dose/11.1` where the real drug starts at about
    `Dose/5.5`, so peak blood concentration after an IV bolus is
    underestimated roughly two-fold. Use this model when the terminal phase
    and total exposure are what matter; use `haloperidol_rat_iv_pk` when the
    peak does.
    """
    return DosedOneCompartmentPK(
        route="IV",
        Vd_L_per_kg=HALOPERIDOL_RAT_IV_VBETA_L_PER_KG,
        CL_mL_per_min_per_kg=HALOPERIDOL_RAT_IV_CL_ML_PER_MIN_PER_KG,
        body_weight_kg=body_weight_kg,
        molar_mass_g_per_mol=molar_mass_g_per_mol,
        name="haloperidol/rat/IV, one-compartment collapse "
        "(Cheng & Paalzow 1992)",
    )


# ---------------------------------------------------------------------------
# Fentanyl
# ---------------------------------------------------------------------------
#
# Sources
# -------
#   [Yassen2008]  Yassen A, Olofsen E, Kan J, Dahan A, Danhof M.
#                "Pharmacokinetic-pharmacodynamic modeling of the
#                effectiveness and safety of buprenorphine and fentanyl in
#                rats." Pharm Res. 2008;25(1):183-93.
#                doi:10.1007/s11095-007-9440-z, PMID 17914664, open access
#                (PMC2190336, read in full). Male Wistar rats (Charles River
#                NL), 225-250 g at the start of the experiment, chronically
#                cannulated (femoral artery for sampling, jugular vein for
#                dosing), conscious. Fentanyl 0.015-0.085 mg/kg given as a
#                zero-order intravenous infusion over 20-40 min; serial
#                arterial PLASMA samples assayed by LC-MS/MS (LLOQ
#                0.118 ng/mL). "For fentanyl, a two-compartment
#                pharmacokinetic model with inter-animal variability on all
#                pharmacokinetic parameters best described the disposition
#                in plasma." NONMEM population estimates (Table II, original
#                data, 95% CI in brackets): Cl 17.0 mL/min [16.9-18.1],
#                V1 250 mL [218-282], V2 440 mL [404-476], Q 9.7 mL/min
#                [8.3-11.1]; the 1,000-replicate bootstrap gives 16.6, 255,
#                444, 9.7. The PK data are pooled from that group's two
#                earlier rat studies (antinociception and respiratory
#                depression), which is why the dose range spans both.
#                What this source does NOT report, checked rather than
#                assumed: no half-life of any phase, and no body-weight
#                normalisation of Cl/V1/V2/Q - the estimates are absolute
#                mL and mL/min for a rat in the 225-250 g band.
#   [CanfieldSprague2025] Canfield JR, Sprague JE. "In vivo pharmacokinetic,
#                pharmacodynamic and brain concentration comparison of
#                fentanyl and para-fluorofentanyl in rats." Arch Toxicol.
#                2025;99(1):287-97. doi:10.1007/s00204-024-03887-z,
#                PMID 39419833, open access (PMC11748481, read in full).
#                Jugular-vein-cannulated male Sprague-Dawley rats (Envigo),
#                n = 6, 326.92 +/- 3.99 g, single 300 ug/kg SUBCUTANEOUS
#                dose, plasma by LC-MS/MS, non-compartmental analysis.
#                Table 1, fentanyl column: t1/2 elimination 1.10 +/- 0.01 h,
#                AUC(0-inf) 112.83 +/- 10.82 ng/mL*h, Cmax 82.12 +/-
#                11.78 ng/mL, Tmax 0.5 h, Clp/F 917.60 +/- 110.11 mL/h,
#                VD/F 1524.90 +/- 368.52 mL (mean +/- SEM). Sampling times
#                were 0, 30, 60, 120, 240 and 480 min, so the reported
#                "Tmax 0.5 h" is the FIRST post-dose sample, not a resolved
#                peak - see SEARCHED_AND_EMPTY for why that makes ka
#                underdetermined.
#   [Almousa2011] Almousa AA, Ikeda R, Wada M, Kuroda N, Hanajiri RK,
#                Nakashima K. "HPLC-UV method development for fentanyl
#                determination in rat plasma and its application to elucidate
#                pharmacokinetic behavior after i.p. administration to rats."
#                J Chromatogr B. 2011;879(27):2941-4.
#                doi:10.1016/j.jchromb.2011.08.029, PMID 21917532; author
#                copy read in full from the Nagasaki University repository
#                (hdl 10069/26235). Male Wistar rats, n = 3, 295-320 g,
#                7-8 weeks; fentanyl 0.1 mg/kg INTRAPERITONEAL; plasma
#                sampled at 0, 10, 15, 20, 25, 30, 40, 90, 120, 180 min;
#                HPLC-UV (LOD 0.8 ng/mL, calibration 5-100 ng/mL). Table 3:
#                Tmax 28.3 +/- 2.9 min, Cmax 21.3 +/- 4.9 ng/mL, T1/2
#                198.3 +/- 44.7 min, AUC(0-180) 15.6e2 +/- 2.9e2 ng*min/mL,
#                MRT(0-180) 75.3 +/- 8.7 min, CL 19.4 +/- 2.6 mL/min (the
#                paper defines its CL as dose/AUC(0-180), i.e. a truncated
#                AUC, and there is no IV arm, so it is an apparent CL/F and
#                an upper bound on one).
#   [SchneiderBrune1985] Schneider E, Brune K. "Distribution of fentanyl in
#                rats: an autoradiographic study." Naunyn Schmiedebergs Arch
#                Pharmacol. 1985;331(4):359-63. doi:10.1007/BF00500820,
#                PMID 4094625. Rat; intragastric administration compared
#                against intravenous: "Its oral bioavailability was found to
#                be only 1.5%." This is the ONLY absolute bioavailability for
#                fentanyl in the rat this project could find by any route.
#   [DaviesMorris1993] as cited in the haloperidol block above; used here
#                only for the same qualitative hepatic-blood-flow comparison.
#   [PubChem3345] PubChem CID 3345, fentanyl, C22H28N2O, molecular weight
#                336.5 g/mol. Chemical fact, used for the mg/L -> nM
#                conversion; the value below is kept byte-identical to
#                `calibration.evidence.FENTANYL_MW` so the two cannot drift.

FENTANYL_MOLAR_MASS_G_PER_MOL = 336.47  # [PubChem3345]

# --- rat, intravenous: the one fully sourced fentanyl parameter set --------
#
# [Yassen2008] reports ABSOLUTE volumes and clearances, not per-kg ones, so
# they are stored absolute here and divided by the caller's body weight at
# construction. That ordering matters: the disposition rate constants
# (k10 = Cl/V1, k12 = Q/V1, k21 = Q/V2) and therefore both half-lives are
# weight-invariant, while the concentration produced by a mg/kg dose scales
# linearly with the weight passed. Storing per-kg constants instead would
# have required picking one weight inside the source's 225-250 g band and
# baking it in invisibly.

#: Plasma clearance, mL/min (absolute) [Yassen2008].
FENTANYL_RAT_IV_CL_ML_PER_MIN = 17.0
#: Central-compartment volume V1, mL (absolute) [Yassen2008].
FENTANYL_RAT_IV_V1_ML = 250.0
#: Peripheral-compartment volume V2, mL (absolute) [Yassen2008].
FENTANYL_RAT_IV_V2_ML = 440.0
#: Inter-compartmental clearance Q, mL/min (absolute) [Yassen2008]. Unlike
#: the haloperidol block, Q here is REPORTED, not derived, so none of that
#: block's ill-conditioning applies to this drug.
FENTANYL_RAT_IV_Q_ML_PER_MIN = 9.7

#: 95% confidence intervals as printed in [Yassen2008] Table II.
FENTANYL_RAT_IV_CL_CI_ML_PER_MIN = (16.9, 18.1)
FENTANYL_RAT_IV_V1_CI_ML = (218.0, 282.0)
FENTANYL_RAT_IV_V2_CI_ML = (404.0, 476.0)
FENTANYL_RAT_IV_Q_CI_ML_PER_MIN = (8.3, 11.1)

#: The 1,000-replicate non-parametric bootstrap estimates from the same
#: table, kept so a test can show the point estimates are not an artefact of
#: a single fit: (Cl, V1, V2, Q) [Yassen2008].
FENTANYL_RAT_IV_BOOTSTRAP = (16.6, 255.0, 444.0, 9.7)

#: Body-weight band of the animals the estimates come from, kg
#: [Yassen2008]: "weighting initially 225-250 g". Not a modeling parameter;
#: recorded so a caller dosing a 500 g rat can see they are extrapolating.
FENTANYL_RAT_IV_SOURCE_BODY_WEIGHT_RANGE_KG = (0.225, 0.250)
#: Dose range over which the model was estimated, mg/kg IV [Yassen2008].
FENTANYL_RAT_IV_SOURCE_DOSE_RANGE_MG_PER_KG = (0.015, 0.085)

#: Bioavailability of the IV route: 1 by definition, not by measurement.
FENTANYL_RAT_IV_F = 1.0

#: Weight used only to instantiate a probe model for the weight-invariant
#: derived quantities below (half-lives, Vz). Any positive weight gives the
#: same answer; `test_fentanyl_pk.py` proves it rather than trusting the
#: claim.
_FENTANYL_RAT_IV_PROBE_BODY_WEIGHT_KG = 0.2375


def fentanyl_rat_iv_pk(
    body_weight_kg, molar_mass_g_per_mol=FENTANYL_MOLAR_MASS_G_PER_MOL
):
    """Two-compartment rat fentanyl PK after an IV dose [Yassen2008]; dose in
    mg/kg.

    This is the primary parameter set for the drug. It is the only fentanyl
    rat/mouse parameter set in reach that is complete (all four disposition
    parameters reported by one study, in one species, from one route, with
    F = 1 by definition), which is why it is the only one this module turns
    into a model.

    The source estimated it from 20-40 min zero-order infusions, not from
    instantaneous boluses. This model dosed as an IV bolus therefore starts
    at Dose/V1 where the real experiment never reached that concentration;
    the disposition afterwards is the source's, the t = 0 intercept is the
    idealisation. `DosedTwoCompartmentPK` has no infusion mode (P1.1 did not
    build one), so this is a limitation to state, not to paper over.

    Concentrations are plasma, in nM by default (pass
    `molar_mass_g_per_mol=None` for mg/L).
    """
    if body_weight_kg is None or body_weight_kg <= 0.0:
        raise ValueError("body_weight_kg must be positive")
    return DosedTwoCompartmentPK(
        route="IV",
        V1_L_per_kg=FENTANYL_RAT_IV_V1_ML / 1000.0 / body_weight_kg,
        V2_L_per_kg=FENTANYL_RAT_IV_V2_ML / 1000.0 / body_weight_kg,
        Q_mL_per_min_per_kg=FENTANYL_RAT_IV_Q_ML_PER_MIN / body_weight_kg,
        CL_mL_per_min_per_kg=FENTANYL_RAT_IV_CL_ML_PER_MIN / body_weight_kg,
        body_weight_kg=body_weight_kg,
        molar_mass_g_per_mol=molar_mass_g_per_mol,
        name="fentanyl/rat/IV (Yassen et al. 2008)",
    )


_FENTANYL_RAT_IV_PROBE = fentanyl_rat_iv_pk(
    _FENTANYL_RAT_IV_PROBE_BODY_WEIGHT_KG, molar_mass_g_per_mol=None
)

#: Terminal (beta) half-life, minutes. DERIVED, not reported: [Yassen2008]
#: publishes a parameter set and no half-life at all, so this is the value
#: its own four numbers imply, obtained from the hybrid rate constants of the
#: two-compartment model. 53.60 min.
FENTANYL_RAT_IV_TERMINAL_HALF_LIFE_MIN = _FENTANYL_RAT_IV_PROBE.terminal_half_life_min
#: Distribution (alpha) half-life, minutes, derived the same way. 5.98 min.
FENTANYL_RAT_IV_DISTRIBUTION_HALF_LIFE_MIN = (
    _FENTANYL_RAT_IV_PROBE.distribution_half_life_min
)
#: Steady-state volume of distribution V1 + V2 = 690 mL (absolute)
#: [Yassen2008].
FENTANYL_RAT_IV_VSS_ML = FENTANYL_RAT_IV_V1_ML + FENTANYL_RAT_IV_V2_ML
#: Terminal-phase volume Vz = CL/beta, mL (absolute), derived. 1314.5 mL.
FENTANYL_RAT_IV_VZ_ML = (
    FENTANYL_RAT_IV_CL_ML_PER_MIN * FENTANYL_RAT_IV_TERMINAL_HALF_LIFE_MIN / LN2
)


def fentanyl_rat_iv_terminal_half_life_interval_min():
    """Interval the source's own 95% CIs leave open for the DERIVED terminal
    half-life: (min, max) over the 16 corners of the (Cl, V1, V2, Q) CI box.

    This is the fentanyl analogue of the haloperidol block's Q-sensitivity
    record, and it exists for the same reason: a derived number should carry
    the uncertainty of the numbers it was derived from, visibly. It is a
    corner enumeration, not a proper joint confidence region (the four
    estimates are correlated and the source does not publish the covariance),
    so it is an outer bound on the uncertainty, and it is used as such: the
    half-life test asserts the exact derived value, and this interval is
    asserted separately so that its width, 44.3-63.6 min, is on the record.
    """
    lows_highs = (
        FENTANYL_RAT_IV_CL_CI_ML_PER_MIN,
        FENTANYL_RAT_IV_V1_CI_ML,
        FENTANYL_RAT_IV_V2_CI_ML,
        FENTANYL_RAT_IV_Q_CI_ML_PER_MIN,
    )
    half_lives = []
    for cl in lows_highs[0]:
        for v1 in lows_highs[1]:
            for v2 in lows_highs[2]:
                for q in lows_highs[3]:
                    model = DosedTwoCompartmentPK(
                        route="IV",
                        V1_L_per_kg=v1 / 1000.0,
                        V2_L_per_kg=v2 / 1000.0,
                        Q_mL_per_min_per_kg=q,
                        CL_mL_per_min_per_kg=cl,
                        body_weight_kg=1.0,
                    )
                    half_lives.append(model.terminal_half_life_min)
    return (min(half_lives), max(half_lives))


def fentanyl_rat_iv_one_compartment_pk(
    body_weight_kg, molar_mass_g_per_mol=FENTANYL_MOLAR_MASS_G_PER_MOL
):
    """One-compartment collapse of [Yassen2008], for continuity with the
    existing single-exponential mu-opioid scenario.

    `Vd = Vz = CL/beta` rather than V1, for the same reason the haloperidol
    collapse uses V_beta: it is the choice that preserves the terminal phase
    (and AUC = Dose/CL, which depends only on clearance). What it gives up is
    the first ten minutes - it starts at Dose/1314 mL where the real drug
    starts near Dose/250 mL, so an IV bolus peak is understated about
    five-fold. That is a much bigger cost than haloperidol's two-fold, which
    is worth stating plainly: for fentanyl the distribution phase is most of
    the story early on, and the two-compartment model is the one to use
    unless only late-time exposure matters.
    """
    if body_weight_kg is None or body_weight_kg <= 0.0:
        raise ValueError("body_weight_kg must be positive")
    return DosedOneCompartmentPK(
        route="IV",
        Vd_L_per_kg=FENTANYL_RAT_IV_VZ_ML / 1000.0 / body_weight_kg,
        CL_mL_per_min_per_kg=FENTANYL_RAT_IV_CL_ML_PER_MIN / body_weight_kg,
        body_weight_kg=body_weight_kg,
        molar_mass_g_per_mol=molar_mass_g_per_mol,
        name="fentanyl/rat/IV, one-compartment collapse (Yassen et al. 2008)",
    )


# --- rat, subcutaneous: sourced disposition, absent absorption -------------
#
# Recorded as data even though no model is built from them, because they are
# real published rat fentanyl numbers and because two of the checks below are
# only possible with them in hand.

#: Reported elimination half-life, minutes: "1.10 +/- 0.01 h"
#: [CanfieldSprague2025].
FENTANYL_RAT_SC_TERMINAL_HALF_LIFE_MIN = 1.10 * 60.0
#: Reported AUC(0-inf), ng/mL*h [CanfieldSprague2025].
FENTANYL_RAT_SC_AUC_NG_H_PER_ML = 112.83
#: Reported Cmax, ng/mL [CanfieldSprague2025].
FENTANYL_RAT_SC_CMAX_NG_PER_ML = 82.12
#: Reported Tmax, h [CanfieldSprague2025]. This equals the first post-dose
#: sampling time; see SEARCHED_AND_EMPTY.
FENTANYL_RAT_SC_TMAX_H = 0.5
#: Reported apparent clearance Clp/F, mL/h (absolute) [CanfieldSprague2025].
FENTANYL_RAT_SC_CL_OVER_F_ML_PER_H = 917.60
#: Reported apparent volume VD/F, mL (absolute) [CanfieldSprague2025].
FENTANYL_RAT_SC_VD_OVER_F_ML = 1524.90
#: Dose and animals the above come from [CanfieldSprague2025].
FENTANYL_RAT_SC_SOURCE_DOSE_MG_PER_KG = 0.300
FENTANYL_RAT_SC_SOURCE_BODY_WEIGHT_KG = 0.32692
#: Blood sampling times, minutes [CanfieldSprague2025].
FENTANYL_RAT_SC_SOURCE_SAMPLING_TIMES_MIN = (0.0, 30.0, 60.0, 120.0, 240.0, 480.0)


def fentanyl_rat_sc_one_compartment_ceiling():
    """Why the SC data cannot be turned into a one-compartment model, as
    arithmetic rather than as a claim.

    For ANY one-compartment model with first-order absorption, the whole
    concentration curve sits under the horizontal line C = F*Dose/V, and
    F*Dose/V = AUC(0-inf) * ke exactly. So `Cmax <= AUC * ke` is a hard
    structural ceiling, independent of ka and of the dose.

    Returns `(cmax, ceiling, ratio)` in ng/mL, all three computed from
    [CanfieldSprague2025]'s own reported means. The ratio is 1.155: the
    reported peak is 15.5% ABOVE the ceiling, so those means are not
    reproducible by a one-compartment model at any ka. That is not an error
    in the paper - it is the signature of a distribution phase, and it agrees
    with [Yassen2008]'s finding that rat fentanyl disposition needs two
    compartments. It does mean the SC route cannot be modelled from this
    source alone, because the source's 30 min first sample cannot resolve
    either the distribution phase or ka.
    """
    ke_per_h = LN2 / (FENTANYL_RAT_SC_TERMINAL_HALF_LIFE_MIN / 60.0)
    ceiling = FENTANYL_RAT_SC_AUC_NG_H_PER_ML * ke_per_h
    return (
        FENTANYL_RAT_SC_CMAX_NG_PER_ML,
        ceiling,
        FENTANYL_RAT_SC_CMAX_NG_PER_ML / ceiling,
    )


def fentanyl_rat_sc_apparent_volume_ml():
    """The two mutually inconsistent readings of [CanfieldSprague2025]'s
    apparent volume, as `(reported, implied_by_CL_and_half_life)` in mL.

    Non-compartmental analysis computes Vz/F as (CL/F)/lambda_z per animal,
    so the three reported means are not independent - yet 917.60/(ln2/1.10 h)
    = 1456 mL, against a reported mean VD/F of 1525 mL, a 4.5% gap. A mean of
    per-animal ratios is not the ratio of the means, which is the whole
    explanation; it is recorded because any later use of these numbers has to
    pick one of the two and should do so knowingly.
    """
    ke_per_h = LN2 / (FENTANYL_RAT_SC_TERMINAL_HALF_LIFE_MIN / 60.0)
    return (
        FENTANYL_RAT_SC_VD_OVER_F_ML,
        FENTANYL_RAT_SC_CL_OVER_F_ML_PER_H / ke_per_h,
    )


# --- rat, intraperitoneal: sourced, and internally inconsistent ------------

#: Reported Tmax, minutes [Almousa2011]. Unlike the SC source's Tmax this one
#: IS resolved: the sampling grid runs 10, 15, 20, 25, 30, 40 min around it.
FENTANYL_RAT_IP_TMAX_MIN = 28.3
#: Reported Cmax, ng/mL [Almousa2011].
FENTANYL_RAT_IP_CMAX_NG_PER_ML = 21.3
#: Reported elimination half-life, minutes [Almousa2011].
FENTANYL_RAT_IP_TERMINAL_HALF_LIFE_MIN = 198.3
#: Reported AUC(0-180), ng*min/mL [Almousa2011]. Truncated at 180 min, not
#: extrapolated to infinity.
FENTANYL_RAT_IP_AUC_0_180_NG_MIN_PER_ML = 15.6e2
#: Reported MRT(0-180), minutes [Almousa2011].
FENTANYL_RAT_IP_MRT_0_180_MIN = 75.3
#: Reported apparent clearance, mL/min (absolute) [Almousa2011], defined by
#: the paper as dose/AUC(0-180).
FENTANYL_RAT_IP_CL_OVER_F_ML_PER_MIN = 19.4
#: Dose and animals [Almousa2011].
FENTANYL_RAT_IP_SOURCE_DOSE_MG_PER_KG = 0.1
FENTANYL_RAT_IP_SOURCE_BODY_WEIGHT_RANGE_KG = (0.295, 0.320)
FENTANYL_RAT_IP_SOURCE_N = 3


def fentanyl_rat_ip_auc_consistency(
    body_weight_kg=sum(FENTANYL_RAT_IP_SOURCE_BODY_WEIGHT_RANGE_KG) / 2.0,
):
    """Check [Almousa2011]'s reported AUC(0-180) against the AUC(0-180) its
    own Cmax, Tmax and T1/2 imply, as `(implied, reported, ratio)` in
    ng*min/mL.

    Construction, all arithmetic on that paper's own numbers:
      * ke = ln2 / T1/2;
      * at the peak of a first-order-absorption one-compartment curve,
        C(tmax) = (F*D/V) * exp(-ke*tmax) exactly, which fixes F*D/V from
        Cmax and Tmax without needing ka;
      * ka solves tmax = ln(ka/ke)/(ka - ke);
      * integrate the resulting Bateman curve from 0 to 180 min.

    The answer is 1.95x the reported AUC(0-180). Both cannot be right. The
    most likely reading is that T1/2 = 198 min is too long: it was fitted on
    tail points at 90-180 min that lie at or below the HPLC-UV assay's 5
    ng/mL lowest calibrator, in n = 3 rats, and it is also ~3x the terminal
    half-life two better-powered rat studies report by other routes (53.6 min
    derived from [Yassen2008] IV, 66 min reported by
    [CanfieldSprague2025] SC). Whatever the cause, a parameter set that
    contradicts itself two-fold is not one to build a model from, so this
    route is recorded and not implemented.
    """
    ke = LN2 / FENTANYL_RAT_IP_TERMINAL_HALF_LIFE_MIN
    fd_over_v = FENTANYL_RAT_IP_CMAX_NG_PER_ML * math.exp(
        ke * FENTANYL_RAT_IP_TMAX_MIN
    )
    # Solve tmax = ln(ka/ke)/(ka-ke) for ka > ke by bisection; the left side
    # is monotonically decreasing in ka, so a plain bracket is enough and it
    # keeps this module free of a scipy dependency.
    lo, hi = ke * (1.0 + 1e-9), 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if math.log(mid / ke) / (mid - ke) > FENTANYL_RAT_IP_TMAX_MIN:
            lo = mid
        else:
            hi = mid
    ka = 0.5 * (lo + hi)
    amplitude = fd_over_v * ka / (ka - ke)
    implied = amplitude * (
        (1.0 - math.exp(-ke * 180.0)) / ke - (1.0 - math.exp(-ka * 180.0)) / ka
    )
    # `body_weight_kg` does not enter: both sides of the comparison are
    # concentrations, and the dose only sets the scale of the apparent
    # volume (see `fentanyl_rat_ip_apparent_volume_ml`). It stays in the
    # signature so the two IP diagnostics are called the same way.
    del body_weight_kg
    return (
        implied,
        FENTANYL_RAT_IP_AUC_0_180_NG_MIN_PER_ML,
        implied / FENTANYL_RAT_IP_AUC_0_180_NG_MIN_PER_ML,
    )


def fentanyl_rat_ip_apparent_volume_ml(
    body_weight_kg=sum(FENTANYL_RAT_IP_SOURCE_BODY_WEIGHT_RANGE_KG) / 2.0,
):
    """Apparent volume V/F implied by [Almousa2011]'s Cmax, Tmax and T1/2,
    in mL: `Dose / (Cmax * exp(ke*Tmax))`.

    4.25 L/kg. Kept because it is the one number from that study that agrees
    with an independent source - [CanfieldSprague2025]'s SC VD/F is 4.45 to
    4.66 L/kg - which is worth recording next to the AUC contradiction so the
    record is "this study is partly corroborated", not "this study is wrong".
    Still not used to build anything: an apparent volume without an F and
    without a trustworthy ke is not a parameter set.
    """
    ke = LN2 / FENTANYL_RAT_IP_TERMINAL_HALF_LIFE_MIN
    dose_ng = FENTANYL_RAT_IP_SOURCE_DOSE_MG_PER_KG * body_weight_kg * 1.0e6
    return dose_ng / (
        FENTANYL_RAT_IP_CMAX_NG_PER_ML * math.exp(ke * FENTANYL_RAT_IP_TMAX_MIN)
    )


# --- rat, oral: the one measured fentanyl bioavailability ------------------

#: Absolute oral (intragastric) bioavailability in the rat, fraction
#: [SchneiderBrune1985]: "Its oral bioavailability was found to be only
#: 1.5%." It is the only absolute F for fentanyl in a rodent this project
#: found by any route, which is why it is recorded. It is NOT enough to build a
#: PO model: that also needs ka, which this source does not report (see
#: SEARCHED_AND_EMPTY), and PO is not a route the behavioural literature
#: this project must fit against uses.
FENTANYL_RAT_PO_F = 0.015


def fentanyl_hepatic_extraction_note(body_weight_kg=0.2375):
    """`(CL_mL_per_min_per_kg, hepatic_blood_flow_mL_per_min_per_kg)` for
    fentanyl in the rat, the same checkable inequality the haloperidol block
    keeps.

    [Yassen2008]'s 17.0 mL/min in a 225-250 g rat is 68-76 mL/min/kg of
    PLASMA clearance, against ~55 mL/min/kg of rat hepatic BLOOD flow
    ([DaviesMorris1993]). Plasma clearance above hepatic blood flow does not
    by itself prove extrahepatic elimination, because the conversion between
    plasma and blood clearance needs a blood:plasma ratio that was searched
    for in the rat and not found (see SEARCHED_AND_EMPTY). What it does
    establish is that fentanyl is a high-extraction drug in the rat, and
    therefore that no extravascular bioavailability for it may be assumed
    near 1 - consistent with the only measured one, 1.5% by the oral route
    ([SchneiderBrune1985]).
    """
    rat_hepatic_blood_flow_mL_per_min_per_kg = 13.8 / 0.25  # [DaviesMorris1993]
    return (
        FENTANYL_RAT_IV_CL_ML_PER_MIN / body_weight_kg,
        rat_hepatic_blood_flow_mL_per_min_per_kg,
    )


# ---------------------------------------------------------------------------
# Searched-and-empty record (the Track E template, applied to PK)
# ---------------------------------------------------------------------------
#
# Kept as data, not only as prose, so that the error a caller hits when they
# ask for an unavailable route is the search record itself.

SEARCHED_AND_EMPTY = {
    ("haloperidol", "rat", "IP"): (
        "No absolute bioavailability F and no absorption rate constant ka "
        "was found for haloperidol given intraperitoneally to rats.\n"
        "Searched (2026-08): PubMed (haloperidol AND rat* AND "
        "pharmacokinetic* AND (subcutaneous OR intraperitoneal OR "
        "bioavailability); 37 hits, none an IP PK study), Europe PMC "
        "including open-access full text (haloperidol AND rats AND "
        "intraperitoneal AND Cmax AND Tmax; hits are nose-to-brain "
        "formulation papers whose comparator arm is IV or intranasal, not a "
        "plain IP solution with a matched IV arm), and web search for a rat "
        "IV-vs-IP crossover.\n"
        "Candidates found and why each is unusable: Pizzolato, Soncrant & "
        "Rapoport, J Neurochem 1984;43:724-32 (PMID 6540296) dosed rats "
        "0.5 and 1 mg/kg i.p. and relates 2-deoxyglucose metabolism to "
        "'estimated brain concentrations', but reports no measured "
        "plasma/blood concentration-time data and no IV arm; Castaing et "
        "al., J Anal Toxicol 2006;30:419-25 (PMID 16959133) dosed 1 mg/kg "
        "i.p. but is a postmortem-redistribution study with no in-life "
        "profile; the nanoparticle/intranasal literature (e.g. the "
        "haloperidol SLN/NLC studies reviewed in PMC8948700) reports brain "
        "targeting ratios against IV or intranasal solution, not IP "
        "bioavailability of the free drug.\n"
        "What is known but deliberately NOT turned into a number: "
        "[ChengPaalzow1992] measured a total blood clearance of "
        "83 mL/min/kg, which is above rat hepatic blood flow "
        "(~55 mL/min/kg, [DaviesMorris1993]); hepatic extraction of the "
        "portal-drained fraction of an IP dose is therefore close to "
        "complete, so F(IP) is certainly well below 1. 'Well below 1' is "
        "not a value, and the extrahepatic component that the >liver-blood-"
        "flow clearance implies makes even the direction of the residual "
        "uncertain. No F is assumed."
    ),
    ("haloperidol", "rat", "SC"): (
        "No absolute bioavailability F and no absorption rate constant ka "
        "was found for haloperidol given subcutaneously to rats.\n"
        "Searched (2026-08): the same PubMed/Europe PMC/web queries as for "
        "IP, plus the D2-occupancy dosing literature that uses the SC route "
        "specifically.\n"
        "Candidates found and why each is unusable: Kapur et al., JPET "
        "2003;305:625-31 (PMID 12606608) is the standard reference for "
        "occupancy-matched haloperidol dosing in rats and does report "
        "'the relationship "
        "between dose, plasma levels, and in vivo dopamine D2 receptor "
        "occupancy', but the full text is closed access (403 from the "
        "publisher; no PMC or repository copy) and its abstract gives only "
        "doses (0.04-0.08 mg/kg single, 0.25 mg/kg/day by pump), names its "
        "routes only as 'daily injections or osmotic minipump infusions', "
        "and reports no PK parameter beyond the qualitative 'the half-life "
        "of antipsychotics in rodents is 4 to 6 times faster than in "
        "humans'; Kapur, Barsoum & Seeman, "
        "Neuropsychopharmacology 2000, gives an SC dose-occupancy curve "
        "(ED50 0.02 mg/kg s.c.) but occupancy is not concentration and the "
        "paper reports no SC plasma profile; the Groningen PK-PD series "
        "(Johnson et al., Pharm Res 2014;31:2605-17, PMID 24792824) fits "
        "rat plasma PK for haloperidol from IV/IP/SC data, but the only "
        "open-access copy is behind a bot-blocked repository endpoint and "
        "the published parameter tables could not be read; quoting them "
        "second-hand would be citing a number nobody in this project has "
        "seen.\n"
        "Note on why F(SC) cannot simply be set to 1: SC absorption avoids "
        "hepatic first pass, but bioavailability is not thereby 1 - it also "
        "depends on absorption from the depot site, which for a lipophilic "
        "base injected as a solution is precisely what the missing ka would "
        "describe. Setting F = 1 would be inventing the number that most "
        "changes achieved concentration."
    ),
    ("fentanyl", "rat", "SC"): (
        "No absorption rate constant ka and no absolute bioavailability F "
        "was found for fentanyl given subcutaneously to rats. The "
        "DISPOSITION side of this route is sourced (Canfield & Sprague, Arch "
        "Toxicol 2025;99:287-97, PMID 39419833, read in full: t1/2 1.10 h, "
        "AUC(0-inf) 112.83 ng/mL*h, Cmax 82.12 ng/mL, Clp/F 917.60 mL/h, "
        "VD/F 1524.90 mL, in six 327 g male Sprague-Dawley rats at "
        "300 ug/kg s.c.), and those numbers are recorded as constants; what "
        "is missing is everything about the absorption phase.\n"
        "Searched (2026-08): PubMed (fentanyl[Title] AND pharmacokinetic* "
        "AND rat, 40 hits reviewed one by one; fentanyl AND bioavailability "
        "AND (rat OR rats OR mice), 9 hits), Europe PMC including "
        "open-access full text for rat/mouse fentanyl PK with Cmax/Tmax/"
        "clearance, and Unpaywall on every closed candidate.\n"
        "Candidates found and why each is unusable: Canfield & Sprague 2025 "
        "sampled at 0, 30, 60, 120, 240 and 480 min, so its reported "
        "'Tmax 0.5 h' is the FIRST post-dose sample rather than a resolved "
        "peak - the true peak is somewhere in (0, 30] min and ka is bounded, "
        "not measured; the same study has no IV arm, so its Clp/F and VD/F "
        "are apparent and carry no F. Its reported means are also "
        "incompatible with a one-compartment absorption model at any ka "
        "(Cmax exceeds the structural ceiling AUC*ke by 15.5%, see "
        "`fentanyl_rat_sc_one_compartment_ceiling`), which is consistent "
        "with the two-compartment disposition Yassen et al. 2008 found by "
        "the IV route but means a two-compartment SC model would need a "
        "distribution phase this dataset cannot resolve either. Canfield, "
        "Fry & Sprague, Drug Alcohol Depend 2025;272:112710 "
        "(PMID 40382854) doses the same route at 50-200 ug/kg but reports "
        "the same apparent parameters with no IV comparator. Kendall et "
        "al., JAALAS 2014;53(5):478-84 (PMC4181689) is the only other "
        "rodent study found reporting a subcutaneous fentanyl "
        "concentration-time profile, and it dosed a sustained-release "
        "formulation, whose half-life is set by the formulation's release "
        "rate rather than by fentanyl's disposition; the paper states "
        "outright that plain fentanyl was not assessed.\n"
        "Why F(SC) is not simply set to 1: subcutaneous dosing bypasses "
        "hepatic first pass, but fentanyl is a high-extraction drug in the "
        "rat (see `fentanyl_hepatic_extraction_note`) and the only measured "
        "rat bioavailability for it, 1.5% by the oral route "
        "(Schneider & Brune 1985), shows how far from 1 an extravascular "
        "route can be for this compound. Setting F = 1 would invent the "
        "single number that most changes the achieved concentration."
    ),
    ("fentanyl", "rat", "IP"): (
        "No absorption rate constant ka and no absolute bioavailability F "
        "was found for fentanyl given intraperitoneally to rats, and the one "
        "published IP parameter set contradicts itself.\n"
        "Searched (2026-08): the same PubMed/Europe PMC/Unpaywall sweep as "
        "for SC, plus a specific search for a rat IV-versus-IP fentanyl "
        "crossover.\n"
        "Candidates found and why each is unusable: Almousa et al., J "
        "Chromatogr B 2011;879(27):2941-4 (PMID 21917532; author copy read in "
        "full from the Nagasaki University repository) is a real rat IP "
        "fentanyl PK study - 0.1 mg/kg i.p., male Wistar, plasma at ten time "
        "points - and its Tmax of 28.3 min IS resolved, so it is the only "
        "rodent fentanyl absorption datum in reach. But it has no IV arm, so "
        "no F; its clearance is defined as dose/AUC(0-180), a truncated AUC, "
        "so it is an upper bound on an apparent clearance rather than a "
        "clearance; and its own three reported quantities do not agree with "
        "each other - the AUC(0-180) implied by its Cmax, Tmax and T1/2 is "
        "1.95x the AUC(0-180) it reports (see "
        "`fentanyl_rat_ip_auc_consistency`). Its T1/2 of 198 min was fitted "
        "on 90-180 min points at or below the assay's 5 ng/mL lowest "
        "calibrator in n = 3 rats, and is ~3x the terminal half-life two "
        "better-powered rat studies give (53.6 min derived from Yassen et "
        "al. 2008 by IV, 66 min reported by Canfield & Sprague 2025 by SC). "
        "Immelman et al., Can J Anaesth 1990;37(1):63-8 (PMID 2295107) reports "
        "serum fentanyl concentrations after intraperitoneal dosing in rats "
        "but as a comparator arm for extradural and subdural analgesia, with "
        "no PK parameters and no IV arm. Yu et al., Toxicol Sci "
        "2018;164:353-62 (PMID 29669042) gives rat brain-versus-plasma "
        "fentanyl exposure but only after intravenous dosing.\n"
        "What is recorded instead of a model: every number Almousa et al. "
        "report is stored as a constant here, together with the two "
        "arithmetic checks above, so the contradiction is inspectable rather "
        "than inherited."
    ),
    ("fentanyl", "rat", "PO"): (
        "F for this route IS sourced - Schneider & Brune, Naunyn "
        "Schmiedebergs Arch Pharmacol 1985;331:359-63 (PMID 4094625) "
        "measured intragastric against intravenous fentanyl in the rat and "
        "report 1.5% oral bioavailability, stored here as "
        "`FENTANYL_RAT_PO_F` - but ka is not. That paper is a whole-body "
        "autoradiography study; it reports the bioavailability figure and no "
        "absorption-phase parameter, and no other rat oral fentanyl "
        "concentration-time dataset was found (searched 2026-08, same "
        "sweep as the SC record). A route needs both numbers, so no PO "
        "model is built. This route is recorded for completeness rather "
        "than need: the rodent behavioural pharmacology this project must "
        "fit against does not dose fentanyl orally."
    ),
    ("fentanyl", "mouse", "*"): (
        "No pharmacokinetic parameter set for fentanyl in the MOUSE was "
        "found for any route: no Vd, no CL, no terminal half-life, no F.\n"
        "Searched (2026-08): PubMed (fentanyl AND (mice OR mouse) AND "
        "pharmacokinetic* AND (serum OR plasma OR clearance OR half-life), "
        "11 hits, each abstract read; fentanyl AND (mice[MeSH] OR mouse OR "
        "mice) AND (pharmacokinetic* OR clearance OR half-life), 44 hits "
        "screened by title), Europe PMC restricted to open access for "
        "mouse fentanyl noncompartmental/Cmax/AUC/half-life full text, "
        "Unpaywall on every closed candidate, and a web search for a "
        "repository or thesis copy of the one canonical study.\n"
        "Candidates found and why each is unusable: Kalvass JC, Olson ER, "
        "Cassidy MP, Selley DE, Pollack GM. J Pharmacol Exp Ther "
        "2007;323:346-55 (PMID 17646430, doi:10.1124/jpet.107.119560) is "
        "the canonical source - seven opioids including fentanyl, "
        "equipotent subcutaneous doses in mice, serum and brain "
        "concentration time courses, unbound fractions - and it is closed "
        "access (Unpaywall is_oa false, no repository copy; the publisher "
        "returns 403 to the article page and to both full-text PDF paths; "
        "no PMC deposit). Its abstract reports no PK parameter. A future "
        "pass with journal access should read this paper first; it would "
        "close this gap and probably P1.5's mouse Kp,uu as well. "
        "Kendall LV, Hansen RJ, Dorsey K, Kang S, Lunghofer PJ, Gustafson "
        "DL. J Am Assoc Lab Anim Sci 2014;53(5):478-84 (PMC4181689, read in "
        "full) reports mouse subcutaneous fentanyl PK, but only for a "
        "sustained-release formulation: female CD1 mice 20-27 g, 3.5 mg/kg "
        "SC interscapular, Tmax 2 h, Cmax 62.1 ng/mL, t1/2 6.28 h, "
        "AUC(inf) 288 h*ng/mL, Cl/Fpred 12.2 L/h/kg. Those numbers describe "
        "the formulation's release rate rather than fentanyl's disposition, "
        "and the paper says so directly: 'Due to the short duration of "
        "action of Fent and Butp, only the SR formulations were assessed.' "
        "Shao et al., Drug Metab Dispos 2026;54(7):100319 (PMID 42424960) "
        "measured mouse fentanyl PK after an intravenous dose as the "
        "anchor for a cross-species PBPK-PD model, and is closed access "
        "(Unpaywall is_oa false). The mouse fentanyl-analogue literature "
        "(Bilel et al. on isobutyrylfentanyl and 4F-furanylfentanyl; "
        "Chen et al., Int J Mol Sci 2026 on 25 analogues) reports PK for "
        "the ANALOGUES, and substituting an analogue's disposition for "
        "fentanyl's is exactly the cross-drug substitution this codebase "
        "forbids. The mouse anti-fentanyl vaccine and monoclonal-antibody "
        "literature reports serum and brain fentanyl concentrations under "
        "antibody sequestration, which is a perturbed system, and reports "
        "PK parameters for the antibody rather than for fentanyl.\n"
        "So this module models fentanyl in the RAT only, and "
        "`fentanyl_mouse_pk` raises rather than scaling the rat parameters "
        "allometrically. Allometry across a single 10x body-mass step, with "
        "no mouse datum to check it against, would produce a number with a "
        "citation attached to it that nobody measured."
    ),
    ("fentanyl", "rat", "blood:plasma"): (
        "No measured rat blood:plasma concentration ratio (Rb) was found for "
        "fentanyl. [Yassen2008] measured plasma, so this parameter set "
        "produces PLASMA concentrations, which is the same matrix as the "
        "occupancy anchor the calibrated mu-opioid scenario already uses "
        "(Moss et al. 2020, in ng/mL of plasma), so nothing in this codebase "
        "currently needs Rb. It is recorded because it is what would be "
        "needed to say whether the plasma clearance being above rat hepatic "
        "blood flow implies extrahepatic elimination, and that question is "
        "left open rather than answered by assumption."
    ),
    ("haloperidol", "rat", "blood:plasma"): (
        "No measured rat blood:plasma concentration ratio (Rb) was found "
        "for haloperidol. [ChengPaalzow1992] reports blood clearance and "
        "blood-based volumes, so this parameter set produces whole-blood "
        "concentrations, while the occupancy anchor this codebase already "
        "uses (Kapur et al. 1997, see scenarios.py) is a human PLASMA "
        "concentration. Converting between them needs Rb, which was "
        "searched for and not found for rat. The nearest datum is human: "
        "plasma and whole-blood haloperidol concentrations were not "
        "significantly different in patients on depot haloperidol, i.e. "
        "Rb ~ 1 in humans, with the reduced-haloperidol metabolite (not "
        "modeled here) behaving very differently (blood:plasma ~3.6). That "
        "is a different species and is recorded as context only; no Rb is "
        "applied, and any comparison of this model's output against a "
        "plasma-based measurement inherits that unquantified gap."
    ),
}


def haloperidol_rat_pk(
    route, body_weight_kg, molar_mass_g_per_mol=HALOPERIDOL_MOLAR_MASS_G_PER_MOL
):
    """Rat haloperidol PK model for `route`.

    IV returns the two-compartment [ChengPaalzow1992] model. IP, SC and PO
    raise `UnsourcedParameterError` carrying the searched-and-empty record
    for that route: those routes need an F and a ka that this project
    searched for and did not find, and the alternative to raising is
    inventing them.
    """
    route = str(route).upper()
    if route == "IV":
        return haloperidol_rat_iv_pk(
            body_weight_kg=body_weight_kg, molar_mass_g_per_mol=molar_mass_g_per_mol
        )
    record = SEARCHED_AND_EMPTY.get(("haloperidol", "rat", route))
    if record is None:
        record = (
            "No parameter set for route %r has been sourced for haloperidol "
            "in the rat, and no search for one has been recorded." % route
        )
    raise UnsourcedParameterError(
        "haloperidol / rat / %s: %s" % (route, record)
    )


def fentanyl_rat_pk(
    route, body_weight_kg, molar_mass_g_per_mol=FENTANYL_MOLAR_MASS_G_PER_MOL
):
    """Rat fentanyl PK model for `route`.

    IV returns the two-compartment [Yassen2008] model. SC, IP and PO raise
    `UnsourcedParameterError` carrying the searched-and-empty record for that
    route. Those routes are not empty in the same way: SC has published
    disposition and no absorption, IP has a published but self-contradictory
    parameter set, and PO has a measured F and no absorption. Each record
    says which, because "not available" and "available and unusable" call for
    different next steps.
    """
    route = str(route).upper()
    if route == "IV":
        return fentanyl_rat_iv_pk(
            body_weight_kg=body_weight_kg, molar_mass_g_per_mol=molar_mass_g_per_mol
        )
    record = SEARCHED_AND_EMPTY.get(("fentanyl", "rat", route))
    if record is None:
        record = (
            "No parameter set for route %r has been sourced for fentanyl in "
            "the rat, and no search for one has been recorded." % route
        )
    raise UnsourcedParameterError("fentanyl / rat / %s: %s" % (route, record))


def fentanyl_mouse_pk(route, body_weight_kg, molar_mass_g_per_mol=None):
    """Always raises `UnsourcedParameterError`: no route has a mouse fentanyl
    parameter set, and the rat one is not scaled into the mouse.

    The signature matches `fentanyl_rat_pk` on purpose, so that a caller who
    writes mouse code gets the search record rather than an AttributeError
    and a temptation to reach for the rat function.
    """
    del body_weight_kg, molar_mass_g_per_mol  # nothing to build them into
    raise UnsourcedParameterError(
        "fentanyl / mouse / %s: %s"
        % (str(route).upper(), SEARCHED_AND_EMPTY[("fentanyl", "mouse", "*")])
    )


def fentanyl_rat_iv_terminal_half_life_min():
    """Terminal half-life of the primary (two-compartment) parameter set, in
    minutes, computed from the model rather than restated.

    Exists for the same reason as its haloperidol counterpart, and matters
    more here: [Yassen2008] reports no half-life at all, so this number is
    only ever a property of the model built from that paper's four
    parameters, and it should be read off the model.
    """
    model = fentanyl_rat_iv_pk(body_weight_kg=0.2375, molar_mass_g_per_mol=None)
    return model.terminal_half_life_min


def haloperidol_rat_iv_terminal_half_life_min():
    """Terminal half-life of the primary (two-compartment) parameter set,
    in minutes, computed from the model rather than restated.

    Exists so tests and P1.6 compare the *model's* half-life against
    [ChengPaalzow1992]'s reported one instead of comparing a constant to
    itself.
    """
    model = haloperidol_rat_iv_pk(body_weight_kg=0.25, molar_mass_g_per_mol=None)
    return model.terminal_half_life_min


def hepatic_extraction_is_high_note():
    """One derived comparison worth keeping addressable in code: rat blood
    clearance vs rat hepatic blood flow.

    Returns `(CL_mL_per_min_per_kg, hepatic_blood_flow_mL_per_min_per_kg)`.
    The clearance exceeding liver blood flow is why no extravascular
    bioavailability for this drug can be assumed close to 1, and also why
    elimination cannot be purely hepatic. Both numbers are cited
    ([ChengPaalzow1992], [DaviesMorris1993]); nothing is inferred from them
    here beyond the inequality itself.
    """
    rat_hepatic_blood_flow_mL_per_min_per_kg = 13.8 / 0.25  # [DaviesMorris1993]
    return (
        HALOPERIDOL_RAT_IV_CL_ML_PER_MIN_PER_KG,
        rat_hepatic_blood_flow_mL_per_min_per_kg,
    )


__all__ = [
    "UnsourcedParameterError",
    "SEARCHED_AND_EMPTY",
    "HALOPERIDOL_MOLAR_MASS_G_PER_MOL",
    "HALOPERIDOL_RAT_IV_CL_ML_PER_MIN_PER_KG",
    "HALOPERIDOL_RAT_IV_VC_L_PER_KG",
    "HALOPERIDOL_RAT_IV_VSS_L_PER_KG",
    "HALOPERIDOL_RAT_IV_VBETA_L_PER_KG",
    "HALOPERIDOL_RAT_IV_V2_L_PER_KG",
    "HALOPERIDOL_RAT_IV_Q_ML_PER_MIN_PER_KG",
    "HALOPERIDOL_RAT_IV_Q_SENSITIVITY_ML_PER_MIN_PER_KG",
    "HALOPERIDOL_RAT_IV_BETA_PER_MIN",
    "HALOPERIDOL_RAT_IV_TERMINAL_HALF_LIFE_MIN",
    "HALOPERIDOL_RAT_IV_TERMINAL_HALF_LIFE_INTERVAL_MIN",
    "HALOPERIDOL_RAT_IV_F",
    "haloperidol_rat_pk",
    "haloperidol_rat_iv_pk",
    "haloperidol_rat_iv_one_compartment_pk",
    "haloperidol_rat_iv_terminal_half_life_min",
    "hepatic_extraction_is_high_note",
    "FENTANYL_MOLAR_MASS_G_PER_MOL",
    "FENTANYL_RAT_IV_CL_ML_PER_MIN",
    "FENTANYL_RAT_IV_V1_ML",
    "FENTANYL_RAT_IV_V2_ML",
    "FENTANYL_RAT_IV_Q_ML_PER_MIN",
    "FENTANYL_RAT_IV_CL_CI_ML_PER_MIN",
    "FENTANYL_RAT_IV_V1_CI_ML",
    "FENTANYL_RAT_IV_V2_CI_ML",
    "FENTANYL_RAT_IV_Q_CI_ML_PER_MIN",
    "FENTANYL_RAT_IV_BOOTSTRAP",
    "FENTANYL_RAT_IV_SOURCE_BODY_WEIGHT_RANGE_KG",
    "FENTANYL_RAT_IV_SOURCE_DOSE_RANGE_MG_PER_KG",
    "FENTANYL_RAT_IV_F",
    "FENTANYL_RAT_IV_VSS_ML",
    "FENTANYL_RAT_IV_VZ_ML",
    "FENTANYL_RAT_IV_TERMINAL_HALF_LIFE_MIN",
    "FENTANYL_RAT_IV_DISTRIBUTION_HALF_LIFE_MIN",
    "FENTANYL_RAT_SC_TERMINAL_HALF_LIFE_MIN",
    "FENTANYL_RAT_SC_AUC_NG_H_PER_ML",
    "FENTANYL_RAT_SC_CMAX_NG_PER_ML",
    "FENTANYL_RAT_SC_TMAX_H",
    "FENTANYL_RAT_SC_CL_OVER_F_ML_PER_H",
    "FENTANYL_RAT_SC_VD_OVER_F_ML",
    "FENTANYL_RAT_SC_SOURCE_DOSE_MG_PER_KG",
    "FENTANYL_RAT_SC_SOURCE_BODY_WEIGHT_KG",
    "FENTANYL_RAT_SC_SOURCE_SAMPLING_TIMES_MIN",
    "FENTANYL_RAT_IP_TMAX_MIN",
    "FENTANYL_RAT_IP_CMAX_NG_PER_ML",
    "FENTANYL_RAT_IP_TERMINAL_HALF_LIFE_MIN",
    "FENTANYL_RAT_IP_AUC_0_180_NG_MIN_PER_ML",
    "FENTANYL_RAT_IP_MRT_0_180_MIN",
    "FENTANYL_RAT_IP_CL_OVER_F_ML_PER_MIN",
    "FENTANYL_RAT_IP_SOURCE_DOSE_MG_PER_KG",
    "FENTANYL_RAT_IP_SOURCE_BODY_WEIGHT_RANGE_KG",
    "FENTANYL_RAT_IP_SOURCE_N",
    "FENTANYL_RAT_PO_F",
    "fentanyl_rat_pk",
    "fentanyl_mouse_pk",
    "fentanyl_rat_iv_pk",
    "fentanyl_rat_iv_one_compartment_pk",
    "fentanyl_rat_iv_terminal_half_life_min",
    "fentanyl_rat_iv_terminal_half_life_interval_min",
    "fentanyl_rat_sc_one_compartment_ceiling",
    "fentanyl_rat_sc_apparent_volume_ml",
    "fentanyl_rat_ip_auc_consistency",
    "fentanyl_rat_ip_apparent_volume_ml",
    "fentanyl_hepatic_extraction_note",
]
