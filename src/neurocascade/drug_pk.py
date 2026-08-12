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
    diazepam,    rat, IV   -- Klotz et al. 1976 (one-compartment collapse;
                              the source is biexponential but publishes no
                              central volume, so no two-compartment model)
    diazepam,    rat, IP   -- Friedman et al. 1986 (apparent disposition) +
                              Hironaka et al. 1984 (ka); a documented
                              cross-study composite
    diazepam,    rat, PO   -- apparent disposition sourced (Hua et al. 2025)
                              but ka is not identified by it; see
                              SEARCHED_AND_EMPTY
    diazepam,    rat, SC   -- NOT AVAILABLE, see SEARCHED_AND_EMPTY
    diazepam,    mouse, *  -- NOT AVAILABLE for any route, see
                              SEARCHED_AND_EMPTY

Brain partition (P1.5)
----------------------

Receptors sit in brain tissue, so from P1.5 onward each drug also has a
plasma-to-brain partition section at the bottom of this module, and the rat
PK builders attach it by default (`brain_partition=USE_SOURCED_BRAIN_
PARTITION`) so that `engine.Simulation` drives receptor occupancy with an
unbound BRAIN concentration rather than a plasma one. Coverage is uneven and
the unevenness is the finding:

    diazepam,    rat   -- COMPLETE. Kp 4.5 [Friedman1986] + f(u,plasma) 0.196
                          [Friedman1986] + f(u,brain) 0.040 [Gustafsson2019]
                          => Kp,uu 0.918. Attached by default.
    haloperidol, rat   -- Kp 22 [Sunderland1987] only; no rat f(u,brain), so
                          it CANNOT drive occupancy and is not attached.
    fentanyl,    rat   -- Kp ~16 [Canfield2025, read off Fig. 5B]; no rat
                          f(u,brain), same outcome.
    diazepam,    mouse -- COMPLETE ([Doran2005]+[Maurer2005] via [Ma2024]),
    haloperidol, mouse -- COMPLETE (same), both recorded and unused, because
                          there is no mouse PK here to multiply.
    everything,  human -- absent; see
                          SEARCHED_AND_EMPTY_BRAIN_PARTITION[("*","human","*")]
                          for why the human-PK calibrated scenarios stay
                          plasma-driven.

`brain_partition.py` explains why a total Kp is not a substitute for a Kp,uu
and refuses to let one act as one.

Diazepam additionally carries an *active metabolite* record
(`DIAZEPAM_ACTIVE_METABOLITE_LIMITATION` and the functions around it): in the
rat, nordiazepam reaches ~80% of the parent's molar exposure with the same
affinity at the same benzodiazepine site, so a single-ligand diazepam model
understates benzodiazepine-site occupancy. That is recorded as a model
limitation rather than silently ignored.
"""

import math

from .brain_partition import BrainPartition
from .pk import (
    LN2,
    DosedOneCompartmentPK,
    DosedTwoCompartmentPK,
    intercompartmental_clearance_from_terminal_rate,
)


class _SourcedBrainPartitionDefault(object):
    """Sentinel default for the builders' `brain_partition=` argument.

    Three-way rather than two-way on purpose. `None` means "build a
    plasma-driven model, I know what that means"; an explicit
    `BrainPartition` means "use this one"; and this sentinel, the default,
    means "attach the sourced, species-matched partition for this drug if one
    exists and can give an unbound brain concentration, otherwise plasma".

    The default is the attaching one because the pre-P1.5 behavior - driving
    a receptor with a plasma concentration - is the bug P1.5 exists to fix,
    and a default that silently reproduces it would undo the task. Which case
    a given model ended up in is readable off `model.receptor_compartment`
    and `drug_pk.brain_partition_status(drug, species)`.
    """

    def __repr__(self):
        return "USE_SOURCED_BRAIN_PARTITION"


#: See `_SourcedBrainPartitionDefault`.
USE_SOURCED_BRAIN_PARTITION = _SourcedBrainPartitionDefault()


def _resolve_brain_partition(brain_partition, drug, species):
    if isinstance(brain_partition, _SourcedBrainPartitionDefault):
        return brain_partition_for(drug, species, require_receptor_drivable=True)
    return brain_partition


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
    body_weight_kg,
    molar_mass_g_per_mol=HALOPERIDOL_MOLAR_MASS_G_PER_MOL,
    brain_partition=USE_SOURCED_BRAIN_PARTITION,
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
        species="rat",
        brain_partition=_resolve_brain_partition(
            brain_partition, "haloperidol", "rat"
        ),
    )


def haloperidol_rat_iv_one_compartment_pk(
    body_weight_kg,
    molar_mass_g_per_mol=HALOPERIDOL_MOLAR_MASS_G_PER_MOL,
    brain_partition=USE_SOURCED_BRAIN_PARTITION,
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
        species="rat",
        brain_partition=_resolve_brain_partition(
            brain_partition, "haloperidol", "rat"
        ),
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
    body_weight_kg,
    molar_mass_g_per_mol=FENTANYL_MOLAR_MASS_G_PER_MOL,
    brain_partition=USE_SOURCED_BRAIN_PARTITION,
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
        species="rat",
        brain_partition=_resolve_brain_partition(brain_partition, "fentanyl", "rat"),
    )


_FENTANYL_RAT_IV_PROBE = fentanyl_rat_iv_pk(
    _FENTANYL_RAT_IV_PROBE_BODY_WEIGHT_KG,
    molar_mass_g_per_mol=None,
    # Explicitly plasma-only: this probe exists solely to read half-lives and
    # volumes off the model, none of which a brain partition touches, and it
    # runs at import time, before the P1.5 section below is defined.
    brain_partition=None,
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
    body_weight_kg,
    molar_mass_g_per_mol=FENTANYL_MOLAR_MASS_G_PER_MOL,
    brain_partition=USE_SOURCED_BRAIN_PARTITION,
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
        species="rat",
        brain_partition=_resolve_brain_partition(brain_partition, "fentanyl", "rat"),
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
# Diazepam
# ---------------------------------------------------------------------------
#
# Sources
# -------
#   [Klotz1976]  Klotz U, Antonin KH, Bieck PR. "Pharmacokinetics and plasma
#                binding of diazepam in man, dog, rabbit, guinea pig and
#                rat." J Pharmacol Exp Ther. 1976;199(1):67-73, PMID 789856
#                (Elsevier doi 10.1016/S0022-3565(25)30657-9). Closed access
#                (Unpaywall: is_oa false, no repository copy, publisher
#                returns 403), so the abstract is the whole of what could be
#                read, and it is quoted here rather than paraphrased: "After
#                an intravenous bolus injection plasma levels of D declined
#                biexponentially in all species examined and the data were
#                analyzed according to the two-compartment open model. The
#                binding of D and desmethyldiazepam has been determined at
#                therapeutic concentrations by equilibrium dialysis in man
#                (96.8 and 96.6%, respectively), dog (96.0 and 95.9%),
#                rabbit (89.9 and 94.7%), guinea pig (91.3 and 78.6%) and rat
#                (86.3 and 90.5%). ... T1/2(beta) and the rates of drug
#                clearance were characteristic figures for each species: from
#                1.1 hours and 81.6 ml/min/kg in the rat to 32.9 hours and
#                0.35 ml/min/kg in man, whereas T1/2(alpha), the half-life of
#                distribution, varied only approximately 3-fold (0.3-1.0
#                hours) in the different species."
#                What this source does NOT give, checked rather than
#                assumed: no central volume (Vc/V1), no Vss, no Q, no
#                rat-specific alpha half-life (the 0.3-1.0 h range spans all
#                four animal species), no rat strain/sex/weight, and no IV
#                dose. So the model built from it below is one-compartment,
#                and that is a limitation of the source, not a modeling
#                preference: the source itself says two compartments.
#   [Friedman1986] Friedman H, Abernethy DR, Greenblatt DJ, Shader RI. "The
#                pharmacokinetics of diazepam and desmethyldiazepam in rat
#                brain and plasma." Psychopharmacology (Berl).
#                1986;88(3):267-70. doi:10.1007/BF00180822, PMID 3083446.
#                Closed access (Unpaywall is_oa false; Springer redirects the
#                PDF through an authorization endpoint), so again the
#                abstract is the record. Rats, single 5 mg/kg INTRAPERITONEAL
#                dose of diazepam, four rats sacrificed at each of 5, 15 and
#                30 min and 1, 1.5, 2, 3, 4, 5 and 6 h; plasma and brain
#                assayed for diazepam and desmethyldiazepam. Diazepam
#                "rapidly disappeared from plasma and brain in parallel, with
#                nearly identical overall half-lives of 0.88 and 0.89 h";
#                apparent volume of distribution 19.3 l/kg, apparent total
#                clearance 255 ml/kg/min; free fractions 19.6% (diazepam) and
#                15.8% (desmethyldiazepam); brain:plasma ratios 4.5 +/- 0.1
#                and 3.5 +/- 0.2; desmethyldiazepam half-lives 1.11 h
#                (plasma) and 1.09 h (brain). There is no intravenous arm, so
#                the volume and clearance are apparent (V/F and CL/F) and are
#                named that way below. The source reports no Tmax, no Cmax
#                and no absorption rate constant.
#   [Hironaka1984] Hironaka T, Fuchino K, Fujii T. "Absorption of diazepam
#                and its transfer through the blood-brain barrier after
#                intraperitoneal administration in the rat." J Pharmacol Exp
#                Ther. 1984;229(3):809-15, PMID 6726659. Closed access
#                (Unpaywall is_oa false). Abstract quoted verbatim for the
#                numbers used: "Unlabeled diazepam of 0.9 mg/kg was
#                administered i.p. to the rat of 21 days of age with a trace
#                amount of [N-methyl-3H] diazepam ... The specific rate
#                constants for absorption, distribution (k12 and k21) and
#                elimination were 0.158, 0.015 and 0.136 and 0.013/min,
#                respectively." MeSH records the animals as male, "Rats,
#                Inbred Strains"; the abstract gives no strain name, no body
#                weight and no group size. This is the ONLY measured
#                absorption rate constant for diazepam given
#                intraperitoneally to a rat that this project found, and its
#                animals are 21-day-old weanlings, which is recorded as the
#                central caveat on the composite IP model below.
#   [Hua2025]    Hua A, Wang L, Zhu Y, Zhou Q, Mamun AA, Wang S, Wu F, Xu M.
#                "Effects of Acacetin on the Pharmacokinetics of Diazepam in
#                vivo and in vitro." Iran J Pharm Res. 2025;24(1):e164825.
#                doi:10.5812/ijpr-164825, PMID 41477118, open access
#                (PMC12749239, read in full). The CONTROL arm is a rat oral
#                diazepam PK dataset: six male Sprague-Dawley rats
#                (220 +/- 20 g) given two weeks of vehicle (DMSO/PEG200/
#                water) intraperitoneally, then diazepam 10 mg/kg by oral
#                gavage; plasma at 0.083, 0.25, 0.5, 1, 2, 3, 4, 6 and 8 h;
#                UPLC-MS/MS. No intravenous arm, so its clearance and volume
#                are apparent. Table 1 (diazepam) and Table 2 (nordiazepam)
#                are transcribed below. Two caveats recorded rather than
#                smoothed over: the controls are vehicle-pretreated for two
#                weeks rather than naive, and the terminal-phase estimates
#                carry very large dispersion (t1/2z 1.81 +/- 1.26 h,
#                Vz/F 164.39 +/- 141.97 L/kg).
#   [Gobbi1987]  Gobbi M, Barone D, Mennini T, Garattini S. "Diazepam and
#                desmethyldiazepam differ in their affinities and efficacies
#                at 'central' and 'peripheral' benzodiazepine receptors."
#                J Pharm Pharmacol. 1987;39(5):388-91.
#                doi:10.1111/j.2042-7158.1987.tb03404.x, PMID 2886589. Rat
#                cerebral cortex and cerebellum: diazepam and
#                desmethyldiazepam have EQUAL affinity at the central
#                benzodiazepine receptor, while desmethyldiazepam "acts as a
#                partial agonist" there, which the authors offer as the
#                explanation for its lower activity in functional tests. No
#                numeric Ki appears in the abstract and the full text is
#                closed, so no Ki is recorded here; the equal-affinity and
#                partial-agonist statements are what is used, and they are
#                qualitative statements used qualitatively.
#   [DaviesMorris1993] as cited in the haloperidol block above.
#   [PubChem3016] PubChem CID 3016, diazepam, C16H13ClN2O, molecular weight
#                284.74 g/mol.
#   [PubChem2997] PubChem CID 2997, nordazepam (nordiazepam,
#                desmethyldiazepam), C15H11ClN2O, molecular weight
#                270.71 g/mol. Needed because the metabolite comparison below
#                is only meaningful in molar, not mass, units.

DIAZEPAM_MOLAR_MASS_G_PER_MOL = 284.74  # [PubChem3016]
NORDIAZEPAM_MOLAR_MASS_G_PER_MOL = 270.71  # [PubChem2997]

# --- rat, intravenous: the reference disposition -------------------------

#: Total plasma clearance in the rat, mL/min/kg [Klotz1976].
DIAZEPAM_RAT_IV_CL_ML_PER_MIN_PER_KG = 81.6
#: Terminal (beta) half-life in the rat as reported, minutes: "1.1 hours"
#: [Klotz1976].
DIAZEPAM_RAT_IV_TERMINAL_HALF_LIFE_MIN = 1.1 * 60.0
#: The interval "1.1 h", printed to two significant figures, actually
#: constrains the terminal half-life to. Used as the tolerance in
#: `tests/test_diazepam_pk.py` for the same reason the haloperidol block
#: uses its own: the source's rounding, not a chosen percentage.
DIAZEPAM_RAT_IV_TERMINAL_HALF_LIFE_INTERVAL_MIN = (1.05 * 60.0, 1.15 * 60.0)
#: Distribution (alpha) half-life range, hours, ACROSS ALL FOUR ANIMAL
#: SPECIES in [Klotz1976] ("varied only approximately 3-fold (0.3-1.0 hours)
#: in the different species"). Deliberately not named a rat value: the source
#: does not break it down by species, and picking the middle of a
#: four-species range and calling it the rat's would be inventing a number.
DIAZEPAM_ALPHA_HALF_LIFE_RANGE_ALL_SPECIES_H = (0.3, 1.0)
#: Fraction of diazepam bound to rat plasma protein, [Klotz1976] equilibrium
#: dialysis: 86.3%.
DIAZEPAM_RAT_PLASMA_BOUND_FRACTION = 0.863
#: Same, for desmethyldiazepam (nordiazepam) in the rat: 90.5% [Klotz1976].
NORDIAZEPAM_RAT_PLASMA_BOUND_FRACTION = 0.905
#: Bioavailability of the IV route: 1 by definition, not by measurement.
DIAZEPAM_RAT_IV_F = 1.0

#: Terminal disposition rate constant beta, derived from the reported
#: half-life. Both this and `..._VBETA_L_PER_KG` are arithmetic on the two
#: numbers [Klotz1976] prints; no new information enters.
DIAZEPAM_RAT_IV_BETA_PER_MIN = LN2 / DIAZEPAM_RAT_IV_TERMINAL_HALF_LIFE_MIN
#: Terminal-phase volume of distribution V_beta = CL/beta, L/kg, derived.
#: 7.77 L/kg. This is the only volume [Klotz1976] supports: it reports no Vc
#: and no Vss, so the two-compartment model its own data fitted cannot be
#: rebuilt here, and `intercompartmental_clearance_from_terminal_rate` cannot
#: help because that inversion needs V1 and V2 as inputs.
DIAZEPAM_RAT_IV_VBETA_L_PER_KG = DIAZEPAM_RAT_IV_CL_ML_PER_MIN_PER_KG / (
    DIAZEPAM_RAT_IV_BETA_PER_MIN * 1000.0
)


def diazepam_rat_iv_pk(
    body_weight_kg,
    molar_mass_g_per_mol=DIAZEPAM_MOLAR_MASS_G_PER_MOL,
    brain_partition=USE_SOURCED_BRAIN_PARTITION,
):
    """One-compartment rat diazepam PK after an IV bolus [Klotz1976]; dose in
    mg/kg.

    `Vd = V_beta = CL/beta = 7.77 L/kg`, so the reported 1.1 h terminal
    half-life is reproduced by construction and `AUC = Dose/CL` is exact.

    The limitation to state plainly, because it is larger here than for the
    other two drugs' collapses: [Klotz1976] found rat diazepam disposition to
    be *biexponential* and fitted a two-compartment model, but publishes
    neither the central volume nor Vss for any species, so the size of the
    distribution phase is not recoverable. This model therefore starts at
    Dose/V_beta where the real drug starts at Dose/Vc, and Vc < V_beta always,
    so the post-bolus peak is understated by an unknown factor. For
    haloperidol that factor was 2 and for fentanyl 5.3, both computable
    because both sources published a central volume; here it cannot be
    computed at all, only bounded below by 1. Late-time exposure and the
    terminal phase are right; the first minutes after an IV bolus are not,
    and by how much is unknown.

    Concentrations are plasma, in nM by default (pass
    `molar_mass_g_per_mol=None` for mg/L).
    """
    return DosedOneCompartmentPK(
        route="IV",
        Vd_L_per_kg=DIAZEPAM_RAT_IV_VBETA_L_PER_KG,
        CL_mL_per_min_per_kg=DIAZEPAM_RAT_IV_CL_ML_PER_MIN_PER_KG,
        body_weight_kg=body_weight_kg,
        molar_mass_g_per_mol=molar_mass_g_per_mol,
        name="diazepam/rat/IV, one-compartment collapse (Klotz et al. 1976)",
        species="rat",
        brain_partition=_resolve_brain_partition(brain_partition, "diazepam", "rat"),
    )


# --- rat, intraperitoneal: the route the behavioural literature uses -------
#
# This is the route Phase 2 needs. Track E's diazepam target is the elevated
# plus maze at 1-2 mg/kg in rats and mice, and that literature doses
# intraperitoneally. Two different papers are needed to cover it, because no
# single rat study reports both the disposition and the absorption: the model
# built below is therefore an explicit CROSS-STUDY COMPOSITE, and every
# function and constant name says which paper each half came from.

#: Apparent volume of distribution V/F, L/kg [Friedman1986]. Apparent: that
#: study has no IV arm.
DIAZEPAM_RAT_IP_VD_OVER_F_L_PER_KG = 19.3
#: Apparent total clearance CL/F, mL/min/kg [Friedman1986].
DIAZEPAM_RAT_IP_CL_OVER_F_ML_PER_MIN_PER_KG = 255.0
#: Reported plasma half-life, minutes: "0.88 h" [Friedman1986].
DIAZEPAM_RAT_IP_TERMINAL_HALF_LIFE_MIN = 0.88 * 60.0
#: Reported BRAIN half-life, minutes: "0.89 h" [Friedman1986]. Kept because
#: the near-identity of the two is this source's central finding (plasma and
#: brain decline in parallel), and because P1.5 will need it.
DIAZEPAM_RAT_IP_BRAIN_HALF_LIFE_MIN = 0.89 * 60.0
#: Brain:plasma concentration ratio, diazepam [Friedman1986]: 4.5 +/- 0.1.
#: Recorded here, used by P1.5, not by this module.
DIAZEPAM_RAT_IP_BRAIN_PLASMA_RATIO = 4.5
#: Plasma free fraction of diazepam in that study: 19.6% [Friedman1986].
#: Note this disagrees with [Klotz1976]'s 13.7% (100 - 86.3) in the rat; two
#: labs, two equilibrium-dialysis assays, a 43% relative spread. Neither is
#: used to build anything here, and the disagreement is recorded rather than
#: averaged away.
DIAZEPAM_RAT_IP_FREE_FRACTION = 0.196
#: Dose and design of the disposition source [Friedman1986].
DIAZEPAM_RAT_IP_SOURCE_DOSE_MG_PER_KG = 5.0
DIAZEPAM_RAT_IP_SOURCE_N_PER_TIMEPOINT = 4
DIAZEPAM_RAT_IP_SOURCE_SAMPLING_TIMES_MIN = (
    5.0,
    15.0,
    30.0,
    60.0,
    90.0,
    120.0,
    180.0,
    240.0,
    300.0,
    360.0,
)

#: First-order absorption rate constant after an IP dose, min^-1
#: [Hironaka1984]. The only measured one found for this drug/species/route.
DIAZEPAM_RAT_IP_KA_PER_MIN = 0.158
#: [Hironaka1984]'s own two-compartment disposition constants, min^-1, kept
#: so the transplant of its ka into [Friedman1986]'s one-compartment
#: disposition can be checked rather than asserted (see
#: `diazepam_rat_ip_composite_disposition_cross_check`).
DIAZEPAM_RAT_IP_KA_SOURCE_K12_PER_MIN = 0.015
DIAZEPAM_RAT_IP_KA_SOURCE_K21_PER_MIN = 0.136
DIAZEPAM_RAT_IP_KA_SOURCE_KEL_PER_MIN = 0.013
#: Age and dose of the ka source's animals [Hironaka1984]: 21-day-old
#: (weanling) male rats at 0.9 mg/kg IP, against [Friedman1986]'s 5 mg/kg.
#: [Friedman1986]'s abstract does not state its animals' age at all, so what
#: is certain is only that the ka half of the composite is measured in
#: weanlings; whether the disposition half is adult is an assumption of the
#: usual kind about an unstated detail. This mismatch is the main caveat on
#: the composite and is stored as data so a test can assert it is still on
#: the record.
DIAZEPAM_RAT_IP_KA_SOURCE_AGE_DAYS = 21
DIAZEPAM_RAT_IP_KA_SOURCE_DOSE_MG_PER_KG = 0.9


def diazepam_rat_ip_pk(
    body_weight_kg,
    molar_mass_g_per_mol=DIAZEPAM_MOLAR_MASS_G_PER_MOL,
    brain_partition=USE_SOURCED_BRAIN_PARTITION,
):
    """Rat diazepam PK after an IP dose; dose in mg/kg. A documented
    cross-study composite, not a single-paper parameter set.

    Disposition is [Friedman1986]'s (V/F = 19.3 L/kg, CL/F = 255 mL/min/kg,
    from 5 mg/kg IP in rats of unstated age). Absorption is [Hironaka1984]'s
    (ka = 0.158/min, from 0.9 mg/kg IP in 21-day-old rats). No rat study
    found reports both.

    Why it is built as an APPARENT parameterisation (`apparent_parameters=
    True`, i.e. F folded into V/F and CL/F rather than stated separately):
    for an extravascular one-compartment model the observed concentration
    curve depends only on V/F, CL/F and ka, never on F alone, so nothing is
    assumed by not knowing F. What the model consequently cannot report is a
    true Vd, a true CL, or the absorbed fraction. A separately derived
    F(IP) ~ 0.32 does exist here (`diazepam_rat_ip_bioavailability`), but it
    is a cross-study derivation and the model deliberately does not depend
    on it.

    What the composite gets right, and what it does not:

    * The terminal phase is [Friedman1986]'s, and matches its reported 0.88 h
      to 0.64% (see `diazepam_rat_ip_reported_half_life_consistency`).
    * The absorption phase is a transplant. It is defensible on the one
      check available - [Hironaka1984]'s own constants imply a terminal
      half-life of 59.7 min against [Friedman1986]'s 52.8 min and
      [Klotz1976]'s IV 66.0 min, so the three rat studies agree on the
      disposition envelope to within 25%
      (`diazepam_rat_ip_composite_disposition_cross_check`) - but the ka
      itself is from weanlings, and a weanling rat is not a small adult.
    * The composite predicts Tmax = 17.1 min
      (`diazepam_rat_ip_predicted_tmax_min`), while [Friedman1986]'s first
      post-dose sample is at 5 min. Whether its measured curve was already
      falling at 5 min - which would put the true ka in those animals
      several-fold above
      0.158/min - cannot be determined, because that paper is closed access
      and its abstract reports no Tmax. This is the most likely way the model
      is wrong, and it is wrong in a stated direction: absorption too slow,
      so early concentrations too low.

    Concentrations are plasma, in nM by default.
    """
    return DosedOneCompartmentPK(
        route="IP",
        Vd_L_per_kg=DIAZEPAM_RAT_IP_VD_OVER_F_L_PER_KG,
        CL_mL_per_min_per_kg=DIAZEPAM_RAT_IP_CL_OVER_F_ML_PER_MIN_PER_KG,
        body_weight_kg=body_weight_kg,
        ka_per_min=DIAZEPAM_RAT_IP_KA_PER_MIN,
        apparent_parameters=True,
        molar_mass_g_per_mol=molar_mass_g_per_mol,
        name="diazepam/rat/IP, composite (Friedman et al. 1986 disposition, "
        "Hironaka et al. 1984 absorption)",
        species="rat",
        brain_partition=_resolve_brain_partition(brain_partition, "diazepam", "rat"),
    )


def diazepam_rat_ip_reported_half_life_consistency():
    """`(implied_min, reported_min, relative_error)` for [Friedman1986]'s own
    three numbers.

    CL/F divided by V/F is an elimination rate constant, so that study's
    apparent volume and apparent clearance already imply a half-life:
    ln2 / (255/19300) = 52.46 min, against the 52.8 min ("0.88 h") it
    reports. The gap is -0.64%, which is inside what the printed precision of
    19.3 and 255 leaves open, so the set is internally consistent. The real
    error is returned rather than swallowed by a tolerance; the test asserts
    this exact value.
    """
    implied = LN2 / (
        DIAZEPAM_RAT_IP_CL_OVER_F_ML_PER_MIN_PER_KG
        / (DIAZEPAM_RAT_IP_VD_OVER_F_L_PER_KG * 1000.0)
    )
    reported = DIAZEPAM_RAT_IP_TERMINAL_HALF_LIFE_MIN
    return (implied, reported, (implied - reported) / reported)


def diazepam_rat_ip_composite_disposition_cross_check():
    """Terminal half-lives, in minutes, of the three independent rat diazepam
    studies in this record: `(hironaka_1984_ip, friedman_1986_ip,
    klotz_1976_iv)`.

    [Hironaka1984] publishes rate constants rather than a half-life, so its
    entry is derived from its own (k10, k12, k21) via the standard
    two-compartment identities: beta = 0.011601/min, i.e. 59.75 min. The
    three land at 59.7, 52.8 and 66.0 min - a 25% spread across two routes,
    two decades, two age groups and three assays.

    This is the check that licenses the composite in `diazepam_rat_ip_pk`.
    It is a weak check and is presented as one: agreeing on the disposition
    envelope does not prove the absorption constant transfers. It only rules
    out the failure mode where the ka source's animals were eliminating the
    drug so differently that nothing about them transfers.
    """
    k10 = DIAZEPAM_RAT_IP_KA_SOURCE_KEL_PER_MIN
    k12 = DIAZEPAM_RAT_IP_KA_SOURCE_K12_PER_MIN
    k21 = DIAZEPAM_RAT_IP_KA_SOURCE_K21_PER_MIN
    b = k10 + k12 + k21
    root = math.sqrt(max(b * b - 4.0 * k10 * k21, 0.0))
    beta = 0.5 * (b - root)
    return (
        LN2 / beta,
        DIAZEPAM_RAT_IP_TERMINAL_HALF_LIFE_MIN,
        DIAZEPAM_RAT_IV_TERMINAL_HALF_LIFE_MIN,
    )


def diazepam_rat_ip_predicted_tmax_min(body_weight_kg=0.25):
    """Tmax the composite IP model predicts, minutes. 17.1 min.

    Read off the model rather than restated, so it cannot drift from it.
    Body weight does not enter (Tmax depends only on ka and ke); the argument
    exists because the model constructor requires a weight.
    """
    return diazepam_rat_ip_pk(
        body_weight_kg, molar_mass_g_per_mol=None
    ).tmax_min()


def diazepam_rat_ip_bioavailability():
    """Absolute IP bioavailability, DERIVED across two studies:
    `(CL_iv_mL_per_min_per_kg, CL_over_F_ip_mL_per_min_per_kg, F)`.

    F = (AUC_ip/Dose_ip) / (AUC_iv/Dose_iv) is the definition of absolute
    bioavailability, and that ratio is algebraically identical to
    CL_iv / (CL/F)_ip. So 81.6 / 255 = 0.320 is not a new assumption; it is
    the definition applied to two published clearances.

    What it *is* is a cross-study comparison, and it inherits everything that
    differs between [Klotz1976] (rat, IV, unstated strain/dose, 1976,
    GLC-ECD) and [Friedman1986] (rat, 5 mg/kg IP, 1986, four rats per time
    point). Neither reports the other's route. So this number is recorded and
    is deliberately NOT used to build anything: `diazepam_rat_ip_pk` uses the
    apparent parameterisation, which needs no F at all. It is here because
    P1.4 was asked for an F by route, because it is the natural place a later
    pass would start if it needed absorbed dose rather than concentration,
    and because 32% by IP against ~8% by the oral route
    (`diazepam_rat_po_bioavailability`) is a coherent picture of a
    high-extraction drug given by a partly-portal route.
    """
    return (
        DIAZEPAM_RAT_IV_CL_ML_PER_MIN_PER_KG,
        DIAZEPAM_RAT_IP_CL_OVER_F_ML_PER_MIN_PER_KG,
        DIAZEPAM_RAT_IV_CL_ML_PER_MIN_PER_KG
        / DIAZEPAM_RAT_IP_CL_OVER_F_ML_PER_MIN_PER_KG,
    )


# --- rat, oral: sourced, and it does not identify an absorption rate ------

#: [Hua2025] Table 1, control arm, diazepam 10 mg/kg PO. Means +/- SD as
#: printed; the SDs are kept because two of them are the reason no model is
#: built from this route.
DIAZEPAM_RAT_PO_CMAX_UG_PER_L = 57.75
DIAZEPAM_RAT_PO_CMAX_SD_UG_PER_L = 24.81
DIAZEPAM_RAT_PO_TMAX_H = 1.33
DIAZEPAM_RAT_PO_TMAX_SD_H = 0.52
DIAZEPAM_RAT_PO_TERMINAL_HALF_LIFE_H = 1.81
DIAZEPAM_RAT_PO_TERMINAL_HALF_LIFE_SD_H = 1.26
DIAZEPAM_RAT_PO_AUC_0_T_UG_H_PER_L = 172.25
DIAZEPAM_RAT_PO_AUC_0_INF_UG_H_PER_L = 182.02
DIAZEPAM_RAT_PO_CL_OVER_F_L_PER_H_PER_KG = 59.78
DIAZEPAM_RAT_PO_VZ_OVER_F_L_PER_KG = 164.39
DIAZEPAM_RAT_PO_VZ_OVER_F_SD_L_PER_KG = 141.97
DIAZEPAM_RAT_PO_MRT_0_INF_H = 3.11
#: Animals and design [Hua2025].
DIAZEPAM_RAT_PO_SOURCE_DOSE_MG_PER_KG = 10.0
DIAZEPAM_RAT_PO_SOURCE_BODY_WEIGHT_KG = 0.220
DIAZEPAM_RAT_PO_SOURCE_N = 6
DIAZEPAM_RAT_PO_SOURCE_SAMPLING_TIMES_H = (
    0.083,
    0.25,
    0.5,
    1.0,
    2.0,
    3.0,
    4.0,
    6.0,
    8.0,
)

#: [Hua2025] Table 2, nordiazepam measured in the same animals after the same
#: oral diazepam dose. This is the quantitative half of the active-metabolite
#: record below.
NORDIAZEPAM_RAT_PO_CMAX_UG_PER_L = 47.75
NORDIAZEPAM_RAT_PO_TMAX_H = 1.25
NORDIAZEPAM_RAT_PO_TERMINAL_HALF_LIFE_H = 1.97
NORDIAZEPAM_RAT_PO_AUC_0_T_UG_H_PER_L = 127.95
NORDIAZEPAM_RAT_PO_AUC_0_INF_UG_H_PER_L = 139.73


def diazepam_rat_po_apparent_volume_l_per_kg():
    """`(reported, implied_by_CL_over_F_and_half_life)`, L/kg.

    Non-compartmental analysis computes Vz/F as (CL/F)/lambda_z per animal,
    so [Hua2025]'s two numbers are not independent, yet
    59.78 / (ln2/1.81 h) = 156.10 L/kg against a printed 164.39, a 5.0% gap.
    Same cause as the fentanyl SC case: a mean of per-animal ratios is not
    the ratio of the means. Recorded so that any later use picks one
    knowingly.
    """
    return (
        DIAZEPAM_RAT_PO_VZ_OVER_F_L_PER_KG,
        DIAZEPAM_RAT_PO_CL_OVER_F_L_PER_H_PER_KG
        / (LN2 / DIAZEPAM_RAT_PO_TERMINAL_HALF_LIFE_H),
    )


def diazepam_rat_po_clearance_auc_consistency():
    """`(dose_over_AUC, reported_CL_over_F, relative_error)`, L/h/kg.

    For any route, CL/F = Dose / AUC(0-inf) identically. [Hua2025] reports
    both: 10 mg/kg / 182.02 ug/L*h = 54.94 L/h/kg against a printed CL/F of
    59.78, an 8.1% gap, again the mean-of-ratios artefact. 8% is small enough
    that neither number is wrong and large enough that the two cannot both be
    used at once, which is why the derived bioavailability below is reported
    as a range across the two readings rather than as one number.
    """
    dose_over_auc = (
        DIAZEPAM_RAT_PO_SOURCE_DOSE_MG_PER_KG
        * 1000.0
        / DIAZEPAM_RAT_PO_AUC_0_INF_UG_H_PER_L
    )
    return (
        dose_over_auc,
        DIAZEPAM_RAT_PO_CL_OVER_F_L_PER_H_PER_KG,
        (dose_over_auc - DIAZEPAM_RAT_PO_CL_OVER_F_L_PER_H_PER_KG)
        / DIAZEPAM_RAT_PO_CL_OVER_F_L_PER_H_PER_KG,
    )


def diazepam_rat_po_bioavailability():
    """Absolute oral bioavailability, derived the same way as the IP one, and
    over both readings of [Hua2025]'s clearance:
    `(F_from_reported_CL_over_F, F_from_dose_over_AUC)` = (0.082, 0.089).

    ~8-9% oral bioavailability in the rat for a drug whose apparent oral
    clearance (996 mL/min/kg) is eighteen times rat hepatic blood flow
    (~55 mL/min/kg, [DaviesMorris1993]) is the expected shape of the answer,
    and it is the strongest single argument in this block against ever
    setting an extravascular F to 1 for diazepam.
    """
    cl_over_f_reported = (
        DIAZEPAM_RAT_PO_CL_OVER_F_L_PER_H_PER_KG * 1000.0 / 60.0
    )
    dose_over_auc = diazepam_rat_po_clearance_auc_consistency()[0] * 1000.0 / 60.0
    return (
        DIAZEPAM_RAT_IV_CL_ML_PER_MIN_PER_KG / cl_over_f_reported,
        DIAZEPAM_RAT_IV_CL_ML_PER_MIN_PER_KG / dose_over_auc,
    )


def diazepam_rat_po_ka_is_not_identified():
    """Why no PO model is built: the source's own numbers imply two
    absorption rate constants that differ 4.6-fold.

    Returns `(ka_from_tmax, ka_from_cmax_to_auc_ratio, ratio)` in min^-1.
    Both are computed with the same ke = ln2/1.81 h, so the discrepancy is
    not a units artefact:

    * From the reported Tmax alone, solving the Bateman stationary point
      Tmax = ln(ka/ke)/(ka - ke) at Tmax = 1.33 h gives ka = 0.0217/min.
    * From the reported Cmax and AUC alone: for a one-compartment
      extravascular model, F*Dose/V = AUC(0-inf)*ke exactly, and
      Cmax/(F*Dose/V) = (ke/ka)^(ke/(ka-ke)), a function of ka/ke only. The
      reported Cmax is 82.9% of AUC*ke, which needs ka/ke = 15.6, i.e.
      ka = 0.0996/min - and that ka would put Tmax at 29.5 min, not 80 min.

    Unlike the fentanyl SC case, the reported Cmax does NOT exceed the
    structural ceiling AUC*ke, so a one-compartment model is not excluded;
    what is excluded is any single ka reproducing Tmax, Cmax and AUC
    together. The likely cause is ordinary and not a fault in the paper: a
    mean of per-animal Cmax values on a discrete sampling grid is biased
    upward relative to the mean curve's peak, and a mean of per-animal Tmax
    values on the same grid is a coarse statistic. But an absorption constant
    that changes 4.6-fold depending on which reported number it is read from
    is not one to build a model on, so the PO route is recorded and not
    implemented.
    """
    ke = LN2 / (DIAZEPAM_RAT_PO_TERMINAL_HALF_LIFE_H * 60.0)
    tmax = DIAZEPAM_RAT_PO_TMAX_H * 60.0
    # ka from Tmax: bisect on the monotonically decreasing ln(ka/ke)/(ka-ke).
    lo, hi = ke * (1.0 + 1e-12), 10.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if math.log(mid / ke) / (mid - ke) > tmax:
            lo = mid
        else:
            hi = mid
    ka_from_tmax = 0.5 * (lo + hi)
    # ka from Cmax/(AUC*ke): bisect on r = ka/ke in r^(-1/(r-1)), which
    # increases monotonically from 0 towards 1 as r grows.
    target = DIAZEPAM_RAT_PO_CMAX_UG_PER_L / (
        DIAZEPAM_RAT_PO_AUC_0_INF_UG_H_PER_L
        * (LN2 / DIAZEPAM_RAT_PO_TERMINAL_HALF_LIFE_H)
    )
    lo_r, hi_r = 1.0 + 1e-9, 1.0e6
    for _ in range(400):
        mid = math.sqrt(lo_r * hi_r)
        if mid ** (-1.0 / (mid - 1.0)) < target:
            lo_r = mid
        else:
            hi_r = mid
    ka_from_cmax = math.sqrt(lo_r * hi_r) * ke
    return (ka_from_tmax, ka_from_cmax, ka_from_cmax / ka_from_tmax)


def diazepam_hepatic_extraction_note():
    """`(CL_mL_per_min_per_kg, hepatic_blood_flow_mL_per_min_per_kg)` for
    diazepam in the rat, the same checkable inequality the other two blocks
    keep.

    [Klotz1976]'s 81.6 mL/min/kg of PLASMA clearance against ~55 mL/min/kg of
    rat hepatic BLOOD flow ([DaviesMorris1993]). As with fentanyl, this is
    not claimed to prove extrahepatic elimination: converting plasma to blood
    clearance needs a rat blood:plasma ratio that was searched for and not
    found (see SEARCHED_AND_EMPTY). It establishes that diazepam is a
    high-extraction drug in the rat, which is what makes the derived
    F(PO) ~ 8% and F(IP) ~ 32% plausible rather than surprising, and what
    forbids assuming any extravascular F near 1.
    """
    rat_hepatic_blood_flow_mL_per_min_per_kg = 13.8 / 0.25  # [DaviesMorris1993]
    return (
        DIAZEPAM_RAT_IV_CL_ML_PER_MIN_PER_KG,
        rat_hepatic_blood_flow_mL_per_min_per_kg,
    )


# --- the active-metabolite question, answered rather than deferred --------

#: The answer, as data rather than only as prose, so a test can assert it is
#: still on the record and `docs/architecture.md` cannot drift from it.
DIAZEPAM_ACTIVE_METABOLITE_LIMITATION = (
    "Nordiazepam (desmethyldiazepam) DOES matter on the timescale this model "
    "simulates, and this codebase does not represent it. The three facts "
    "that settle it, each from a primary source read for this task:\n"
    "1. EXPOSURE. In the rat, after diazepam 10 mg/kg PO, nordiazepam's "
    "plasma AUC(0-inf) is 139.73 ug/L*h against the parent's 182.02, which "
    "at 270.71 and 284.74 g/mol is 0.516 vs 0.639 umol/L*h, i.e. the "
    "metabolite carries 81% of the parent's MOLAR exposure. Its Tmax "
    "(1.25 h) and terminal half-life (1.97 h) are essentially the parent's "
    "(1.33 h, 1.81 h), so it is not a late accumulation phenomenon that a "
    "short simulation could ignore - it is already there at the peak "
    "([Hua2025], Tables 1 and 2, same animals). By the intraperitoneal "
    "route [Friedman1986] independently finds nordiazepam in rat plasma and "
    "brain with half-lives of 1.11 and 1.09 h against diazepam's 0.88 and "
    "0.89 h, and a brain:plasma ratio of 3.5 against 4.5.\n"
    "2. POTENCY. Nordiazepam is not a bystander at the site this model "
    "occupies: in rat cerebral cortex and cerebellum it has the SAME "
    "affinity as diazepam at the central benzodiazepine receptor, while "
    "behaving as a PARTIAL AGONIST there ([Gobbi1987]). So it competes for "
    "the exact binding site `scenarios.gaba_a_receptor_site_calibrated()` "
    "models, with comparable Ki and lower efficacy.\n"
    "3. TIMESCALE. The existing GABA-A scenario runs to t = 40 min and the "
    "elevated-plus-maze protocols Phase 2 must fit test 30-60 min after an "
    "IP dose. Nordiazepam's rat Tmax is inside that window by both routes "
    "measured.\n"
    "CONSEQUENCE, stated as a limitation and not patched: every diazepam "
    "simulation this codebase produces UNDERSTATES total benzodiazepine-site "
    "occupancy, because roughly 80% more equal-affinity ligand (at reduced "
    "efficacy) is present and unmodelled. The fix is structural, not a "
    "parameter change - a metabolite compartment feeding a second Ligand at "
    "the same ReceptorSite - and it needs two numbers this project does not "
    "have: a rat formation-rate/fraction-metabolised for the demethylation "
    "step (the reported nordiazepam CL/F 85.13 L/h/kg is apparent in BOTH "
    "the parent's F and the formation fraction, so it does not give one), "
    "and a numeric Ki and efficacy for nordiazepam at the benzodiazepine "
    "site ([Gobbi1987]'s equal-affinity and partial-agonist findings are "
    "qualitative in the reachable abstract). Until both exist, no "
    "nordiazepam ligand is added: inventing an efficacy for a partial "
    "agonist is exactly the kind of number this codebase forbids.\n"
    "TEMAZEPAM AND OXAZEPAM: not quantified for the rat by either rat "
    "dataset in this record. [Hua2025] measured diazepam and nordiazepam "
    "only; [Friedman1986] measured diazepam and desmethyldiazepam only. A "
    "rat plasma or brain concentration-time course for temazepam or oxazepam "
    "after diazepam was searched for (August 2026, PubMed and Europe PMC, "
    "the same sweeps recorded in SEARCHED_AND_EMPTY) and not found, so "
    "nothing is asserted about their magnitude in the rat either way. They "
    "are demonstrably measurable in the MOUSE: Takahashi et al., Br J "
    "Pharmacol 2026, doi 10.1111/bph.70430, PMID 42037156, report nordazepam "
    "and oxazepam in midgestational mice after IV diazepam, both exceeding "
    "the parent's fetal Cmax - which is one more reason the mouse is not "
    "modelled from rat numbers."
)


def diazepam_nordiazepam_molar_exposure_ratio():
    """`(diazepam_umol_h_per_L, nordiazepam_umol_h_per_L, ratio)` after
    diazepam 10 mg/kg PO in the rat [Hua2025].

    The comparison has to be molar, not mass: the two compounds differ by a
    methyl group, so a mass-based AUC ratio (139.73/182.02 = 0.768)
    understates the number of metabolite molecules per parent molecule
    (0.807) by about 5%. Small, but it is the sort of unit slip that makes
    an occupancy argument quietly wrong, so the conversion is done in code
    rather than in a comment.
    """
    dz = DIAZEPAM_RAT_PO_AUC_0_INF_UG_H_PER_L / DIAZEPAM_MOLAR_MASS_G_PER_MOL
    ndz = NORDIAZEPAM_RAT_PO_AUC_0_INF_UG_H_PER_L / NORDIAZEPAM_MOLAR_MASS_G_PER_MOL
    return (dz, ndz, ndz / dz)


def diazepam_rat_pk(
    route,
    body_weight_kg,
    molar_mass_g_per_mol=DIAZEPAM_MOLAR_MASS_G_PER_MOL,
    brain_partition=USE_SOURCED_BRAIN_PARTITION,
):
    """Rat diazepam PK model for `route`.

    IV returns the one-compartment collapse of [Klotz1976]; IP returns the
    documented [Friedman1986]+[Hironaka1984] composite. PO and SC raise
    `UnsourcedParameterError` carrying their own search records, and the two
    records are different in kind: PO has a complete apparent parameter set
    that simply does not identify ka, while SC has nothing at all.
    """
    route = str(route).upper()
    if route == "IV":
        return diazepam_rat_iv_pk(
            body_weight_kg=body_weight_kg,
            molar_mass_g_per_mol=molar_mass_g_per_mol,
            brain_partition=brain_partition,
        )
    if route == "IP":
        return diazepam_rat_ip_pk(
            body_weight_kg=body_weight_kg,
            molar_mass_g_per_mol=molar_mass_g_per_mol,
            brain_partition=brain_partition,
        )
    record = SEARCHED_AND_EMPTY.get(("diazepam", "rat", route))
    if record is None:
        record = (
            "No parameter set for route %r has been sourced for diazepam in "
            "the rat, and no search for one has been recorded." % route
        )
    raise UnsourcedParameterError("diazepam / rat / %s: %s" % (route, record))


def diazepam_mouse_pk(route, body_weight_kg, molar_mass_g_per_mol=None):
    """Always raises `UnsourcedParameterError`: no route has a mouse diazepam
    parameter set, and the rat one is not scaled into the mouse.

    Same shape and same reason as `fentanyl_mouse_pk`, with one extra reason
    specific to this drug: rat and mouse are documented to differ in which
    diazepam metabolite dominates, so borrowing rat disposition for the mouse
    would import the wrong metabolic picture as well as the wrong numbers.
    """
    del body_weight_kg, molar_mass_g_per_mol
    raise UnsourcedParameterError(
        "diazepam / mouse / %s: %s"
        % (str(route).upper(), SEARCHED_AND_EMPTY[("diazepam", "mouse", "*")])
    )


def diazepam_rat_iv_terminal_half_life_min():
    """Terminal half-life of the rat IV model, minutes, read off the model.

    66.0 min by construction (Vd = CL/beta), which is the point: it shows the
    collapse preserves [Klotz1976]'s reported 1.1 h exactly rather than
    approximately.
    """
    model = diazepam_rat_iv_pk(body_weight_kg=0.25, molar_mass_g_per_mol=None)
    return model.terminal_half_life_min


def diazepam_rat_ip_terminal_half_life_min():
    """Terminal half-life of the composite rat IP model, minutes, read off
    the model. 52.46 min, against [Friedman1986]'s reported 52.8 min."""
    model = diazepam_rat_ip_pk(body_weight_kg=0.25, molar_mass_g_per_mol=None)
    return model.terminal_half_life_min


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
    ("diazepam", "rat", "PO"): (
        "The oral route's DISPOSITION is sourced and its ABSORPTION is not "
        "identified, which is a different failure from an absent dataset and "
        "is recorded as such.\n"
        "Sourced (Hua A, Wang L, Zhu Y, Zhou Q, Mamun AA, Wang S, Wu F, Xu M. "
        "Iran J Pharm Res 2025;24(1):e164825, doi 10.5812/ijpr-164825, "
        "PMID 41477118, open access PMC12749239, read in full; control arm, "
        "six male Sprague-Dawley rats 220 +/- 20 g, diazepam 10 mg/kg by "
        "gavage, plasma at 0.083-8 h, UPLC-MS/MS): Cmax 57.75 +/- 24.81 "
        "ug/L, Tmax 1.33 +/- 0.52 h, t1/2z 1.81 +/- 1.26 h, AUC(0-inf) "
        "182.02 ug/L*h, CLz/F 59.78 +/- 18.43 L/h/kg, Vz/F 164.39 +/- 141.97 "
        "L/kg, plus the same parameters for nordiazepam. Every one of those "
        "numbers is stored as a constant here.\n"
        "Why no model is built from them: the reported Tmax and the reported "
        "Cmax/AUC ratio imply absorption rate constants that differ 4.6-fold "
        "(0.0217/min vs 0.0996/min, see "
        "`diazepam_rat_po_ka_is_not_identified`), so there is no single ka "
        "reproducing the study's own three headline numbers. Two smaller "
        "internal inconsistencies are recorded alongside it: the printed "
        "Vz/F is 5.0% away from (CL/F)/lambda_z "
        "(`diazepam_rat_po_apparent_volume_l_per_kg`) and the printed CL/F "
        "is 8.1% away from Dose/AUC "
        "(`diazepam_rat_po_clearance_auc_consistency`), both the usual "
        "mean-of-per-animal-ratios artefact. The controls were also "
        "vehicle-pretreated (DMSO/PEG200/water i.p. for two weeks) rather "
        "than naive, which is not obviously neutral for a CYP-cleared drug "
        "and is not something this project can correct for.\n"
        "Also searched (August 2026) for a rat oral diazepam study with a "
        "matched IV arm: PubMed (`diazepam AND rat* AND pharmacokinetic* AND "
        "(intravenous OR clearance OR bioavailability)`; the Europe PMC "
        "abstract-field query `diazepam AND bioavailability AND rat AND "
        "oral`, 9 hits, each screened), Europe PMC title sweep of all rat "
        "diazepam pharmacokinetics/disposition/kinetics papers (8 hits), and "
        "Unpaywall on every closed candidate. The one study that did dose "
        "both routes - Diaz-Garcia JM, Oliver-Botana J, Fos-Galve D, "
        "'Pharmacokinetics of diazepam in the rat: influence of a carbon "
        "tetrachloride-induced hepatic injury', J Pharm Sci 1992;81(8):768-72 "
        "(doi 10.1002/jps.2600810810, PMID 1403721), with its companion "
        "Eur J Drug Metab Pharmacokinet 1991;Spec No 3:94-101 (PMID 1820943) "
        "- gave 4 mg/kg IV and 4 mg/kg oral in normal and cirrhotic rats, "
        "which is exactly the crossover needed, but both are closed access "
        "(Unpaywall is_oa false, no repository copy) and neither abstract "
        "prints a single control-group value: they report only that the "
        "cirrhotic group's half-life and volume rose. That is the first "
        "paper a future pass with journal access should read for this route.\n"
        "What IS derived rather than left blank: F(PO) = 8.2-8.9%, from "
        "CL_iv/(CL/F)_po across [Klotz1976] and [Hua2025], both readings "
        "given by `diazepam_rat_po_bioavailability`."
    ),
    ("diazepam", "rat", "SC"): (
        "No pharmacokinetic dataset of any kind was found for diazepam given "
        "SUBCUTANEOUSLY to a rat: no F, no ka, no Cmax/Tmax, no apparent "
        "clearance or volume.\n"
        "Searched (August 2026): Europe PMC "
        "(`TITLE:\"diazepam\" AND (TITLE:\"subcutaneous\" OR "
        "ABSTRACT:\"subcutaneous\")`, 36 hits, every title screened and the "
        "plausible ones opened; `diazepam AND (rat OR rats) AND "
        "(\"subcutaneous administration\" OR \"subcutaneously\") AND (plasma "
        "OR serum) AND (concentration OR kinetics) AND (bioavailability OR "
        "Cmax OR \"half-life\")`), plus the title/abstract sweeps recorded "
        "for the oral route.\n"
        "Every subcutaneous hit is either a different drug in the same paper "
        "(e.g. rectal diazepam compared against subcutaneous "
        "morphine-scopolamine, PMID 4072591), or uses the word for a "
        "non-diazepam co-treatment, or is a chronic-dependence study dosing "
        "by implanted capsule or minipump - where the release rate of the "
        "implant, not diazepam's absorption, sets the profile, the same "
        "objection that disqualified the sustained-release fentanyl studies "
        "in the fentanyl record above.\n"
        "This route is recorded for completeness rather than need: the "
        "rodent elevated-plus-maze literature this project must fit against "
        "doses diazepam intraperitoneally, and that route IS modelled. "
        "F(SC) is not set to 1 on the grounds that subcutaneous dosing "
        "bypasses hepatic first pass, for the reason the whole block makes "
        "concrete: the one extravascular route with an F derivable here, the "
        "oral one, comes out at 8-9%, and diazepam's rat plasma clearance "
        "exceeds rat hepatic blood flow "
        "(`diazepam_hepatic_extraction_note`)."
    ),
    ("diazepam", "mouse", "*"): (
        "No pharmacokinetic parameter set for diazepam in the MOUSE was "
        "found for any route: no Vd, no CL, no terminal half-life, no F.\n"
        "Searched (August 2026): Europe PMC (`TITLE:\"diazepam\" AND "
        "(TITLE:\"mice\" OR TITLE:\"mouse\")`, 444 hits screened by title; "
        "`diazepam AND (mice OR mouse) AND pharmacokinetics AND (plasma OR "
        "serum) AND (clearance OR half-life)`; `(diazepam) AND (mice OR "
        "mouse) AND (\"volume of distribution\" OR \"elimination half-life\" "
        "OR \"terminal half-life\") AND (intravenous OR intraperitoneal OR "
        "subcutaneous)`, sorted by citation count; `TITLE:\"benzodiazepine\" "
        "AND (mouse OR mice) AND \"pharmacokinetics\" AND diazepam`), PubMed "
        "web search for the classic mouse disposition studies, and Unpaywall "
        "on every closed candidate.\n"
        "Candidates found and why each is unusable: Greenblatt DJ, Sethy VH, "
        "Psychopharmacology (Berl) 1990;102(3):373-8 (doi 10.1007/BF02244106, "
        "PMID 1979181) is the closest thing to the right study - male CF-1 "
        "mice, intravenous diazepam at 3 and 10 umol/kg, plasma AND brain "
        "concentrations from 1 min to 24 h, with ex vivo "
        "3H-flunitrazepam-displacement occupancy in the same animals - but it "
        "is closed access (Springer; no PMC deposit) and its abstract reports "
        "no pharmacokinetic parameter at all, only that diazepam 'entered "
        "brain rapidly, and was also cleared rapidly'. THIS IS THE FIRST "
        "PAPER A FUTURE PASS WITH JOURNAL ACCESS SHOULD READ: it would close "
        "this gap and P1.5's mouse brain:plasma partition together. File SE, "
        "Greenblatt DJ, Martin IL, Brown C, Psychopharmacology 1985;86:137-41 "
        "reports mouse brain diazepam concentrations and receptor occupancy "
        "across strains, but as correlates of anticonvulsant duration, not as "
        "a parameterised time course, and is likewise closed. The "
        "Marcucci/Garattini series (Eur J Pharmacol 1968-70, Br J Pharmacol "
        "1968;34:667P-668P and 1970;39:455P-457P) compared mouse and rat "
        "brain and blood levels after 5 mg/kg i.v. diazepam and is the "
        "origin of the finding that N-desmethyl metabolites accumulate in "
        "mouse brain but not rat brain; the Eur J Pharmacol papers are closed "
        "access and the two Br J Pharmacol items are two- and three-page "
        "scanned meeting proceedings whose text is not machine-readable, so "
        "no number from them is quoted here. Takahashi Y et al., Br J Pharmacol "
        "2026, doi 10.1111/bph.70430, PMID 42037156, "
        "gives intravenous diazepam pharmacokinetics in midgestational "
        "PREGNANT mice with fetal Cmax for diazepam, nordazepam, oxazepam and "
        "oxazepam-glucuronide; pregnancy is a deliberately altered "
        "disposition state and the reported parameters are fetal-transfer "
        "ratios rather than maternal PK, and it is closed access. The "
        "intranasal-formulation mouse literature (e.g. PMC12963071, 2026: "
        "C57 mice, 1 mg/kg intranasal, plasma 49.1 ng/mL and brain "
        "156.3 ng/g at 10 min) reports two-timepoint concentrations in "
        "pilocarpine-treated animals and no PK parameter, and the route is "
        "not one this model has.\n"
        "So `diazepam_mouse_pk` raises for every route and the rat "
        "parameters are NOT scaled allometrically into the mouse. Beyond the "
        "general objection to a single unchecked 10x allometric step, this "
        "drug has a specific one: rat and mouse are documented to differ in "
        "which diazepam metabolite dominates (Marcucci et al., above), so "
        "rat numbers would import the wrong metabolic picture along with the "
        "wrong disposition."
    ),
    ("diazepam", "rat", "blood:plasma"): (
        "No measured rat blood:plasma concentration ratio (Rb) was found for "
        "diazepam. [Klotz1976] states it measured 'diazepam ... in blood and "
        "plasma', which means the ratio was in that paper's reach, but the "
        "abstract reports only the plasma-protein-bound fractions (86.3% "
        "diazepam, 90.5% desmethyldiazepam in the rat) and the full text is "
        "closed. All three diazepam parameter sets recorded here are "
        "therefore PLASMA, which is at least self-consistent, and the "
        "unresolved Rb is why the clearance-versus-hepatic-blood-flow "
        "comparison in `diazepam_hepatic_extraction_note` is left as an "
        "inequality about extraction rather than turned into a claim about "
        "extrahepatic elimination."
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
    route,
    body_weight_kg,
    molar_mass_g_per_mol=HALOPERIDOL_MOLAR_MASS_G_PER_MOL,
    brain_partition=USE_SOURCED_BRAIN_PARTITION,
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
            body_weight_kg=body_weight_kg,
            molar_mass_g_per_mol=molar_mass_g_per_mol,
            brain_partition=brain_partition,
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
    route,
    body_weight_kg,
    molar_mass_g_per_mol=FENTANYL_MOLAR_MASS_G_PER_MOL,
    brain_partition=USE_SOURCED_BRAIN_PARTITION,
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
            body_weight_kg=body_weight_kg,
            molar_mass_g_per_mol=molar_mass_g_per_mol,
            brain_partition=brain_partition,
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


# ===========================================================================
# P1.5: plasma -> brain partition
# ===========================================================================
#
# `brain_partition.py` holds the structure (Kp vs Kp,uu, the refusals, the
# instantaneous-equilibrium assumption); this section holds the measured
# numbers, one drug and one species at a time, in the same style as the PK
# blocks above. Read `brain_partition.py`'s module docstring first: the
# single most important fact here is that Kp and Kp,uu are NOT
# interchangeable, and that only a Kp,uu (or a Kp plus a measured brain
# unbound fraction) can legitimately drive receptor occupancy.
#
# Sources
# -------
#   [Friedman1986]  Already cited in the diazepam block above (Friedman H,
#                Abernethy DR, Greenblatt DJ, Shader RI. "The
#                pharmacokinetics of diazepam and desmethyldiazepam in rat
#                brain and plasma." Psychopharmacology (Berl).
#                1986;88(3):267-70, PMID 3083446). Rat, 5 mg/kg IP; reports
#                a brain:plasma concentration ratio of 4.5 +/- 0.1, a plasma
#                free fraction of 19.6%, and - the reason this study is the
#                right one for a partition rather than only for disposition
#                - brain and plasma half-lives of 0.89 h and 0.88 h, i.e.
#                brain and plasma decline in parallel. That parallel decline
#                is the published evidence licensing the instantaneous-
#                equilibrium assumption for diazepam in the rat.
#   [Gustafsson2019] Gustafsson S, Sehlin D, Lampa E, Hammarlund-Udenaes M,
#                Loryan I. "Heterogeneous drug tissue binding in brain
#                regions of rats, Alzheimer's patients and controls: impact
#                on translational drug development." Sci Rep.
#                2019;9(1):5308. doi:10.1038/s41598-019-41828-4, PMID
#                30926941, open access (PMC6440985), Table 2 read directly.
#                Equilibrium dialysis of brain homogenate with the dilution
#                correction applied, healthy male Sprague-Dawley rats, tissue
#                pooled 2-3 rats per pool. Rat diazepam f(u,brain) by region
#                (means): frontal cortex 0.037, parietal cortex 0.043, basal
#                ganglia 0.044, cerebellum 0.055, hippocampus 0.048, WHOLE
#                BRAIN 0.040 (SD 0.006, N = 2 biological replicates, n = 6
#                technical). Inter-day CV for diazepam 10.3%.
#   [Sunderland1987] Sunderland T, Cohen BM. "Blood to brain distribution of
#                neuroleptics." Psychiatry Res. 1987;20(4):299-305.
#                doi:10.1016/0165-1781(87)90090-4, PMID 2885876. Rat.
#                Abstract quoted verbatim from PubMed: "At doses that led to
#                plasma levels similar to those achieved in clinical use,
#                brain to plasma ratios ranged from 34 and 22 for
#                fluphenazine and haloperidol, respectively, to 2.2 and 0.97
#                for thioridazine and mesoridazine, respectively." Limits,
#                checked rather than guessed: the full text is closed
#                (Unpaywall: is_oa false, no repository copy), so dose,
#                route, timing, strain, sex and n were not obtainable; and
#                the assay is a RADIORECEPTOR assay, which measures
#                neuroleptic-equivalent D2-binding activity rather than
#                chemically specific haloperidol, so any active metabolite
#                contributes to both the numerator and the denominator.
#   [Canfield2025] Already cited in the fentanyl block above (Canfield JR,
#                Sprague JE, Arch Toxicol. 2025;99(1):287-97, PMID 39419833,
#                open access PMC11748481). The same six male Sprague-Dawley
#                rats (355 +/- 2.86 g) dosed 300 ug/kg SC that supply this
#                module's fentanyl SC disposition constants also supply
#                region-resolved brain:plasma ratios at 1 h post-dose,
#                computed per animal as brain concentration (ng/g) divided
#                by that animal's own 1 h plasma concentration (ng/mL).
#                CRITICAL LIMIT ON THIS NUMBER: the paper reports those
#                ratios ONLY graphically, in Fig. 5B; there is no printed
#                value in text, table, or supplement. The constants below are
#                therefore FIGURE READS off the published bar chart, not
#                quoted numbers, and are recorded with an explicit read
#                uncertainty. They are cross-checked against the same
#                figure's panel A (brain concentrations ~1.1-1.3 ug/g) and
#                the paper's own plasma parameters, which agree; see
#                `fentanyl_rat_brain_plasma_figure_read_cross_check`.
#   [HugMurphy1981] Hug CC Jr, Murphy MR. "Tissue redistribution of fentanyl
#                and termination of its effects in rats." Anesthesiology.
#                1981;55(4):369-75. doi:10.1097/00000542-198110000-00006,
#                PMID 7294371. Rats given 50 ug/kg IV [3H]fentanyl,
#                sacrificed at 1.5-240 min. Abstract quoted verbatim:
#                "Fentanyl concentrations in brain, heart, and lung
#                equilibrated with that in plasma before 1.5 min and declined
#                at the same rate (t 1/2 alpha = 8 min, t 1/2 beta = 45
#                min)." Used ONLY for that qualitative statement, which is
#                the published evidence licensing the instantaneous-
#                equilibrium assumption for fentanyl in the rat. Unpaywall
#                reports this as open access, but the publisher's PDF
#                endpoint (pubs.asahq.org) returns 403 to every automated
#                request tried, so the tissue:plasma ratio tables in the full
#                text could not be read and no number is taken from them.
#   [Doran2005]  Doran A, Obach RS, Smith BJ, et al. "The impact of
#                P-glycoprotein on the disposition of drugs targeted for
#                indications of the central nervous system: evaluation using
#                the MDR1A/1B knockout mouse model." Drug Metab Dispos.
#                2005;33(1):165-74. doi:10.1124/dmd.104.001230,
#                PMID 15502009. MOUSE. Source of the mouse Kp,brain values
#                below.
#   [Maurer2005] Maurer TS, DeBartolo DB, Tess DA, Scott DO. "Relationship
#                between exposure and nonspecific binding of thirty-three
#                central nervous system drugs in mice." Drug Metab Dispos.
#                2005;33(1):175-81. doi:10.1124/dmd.104.001222,
#                PMID 15502010. MOUSE, companion paper to [Doran2005] in the
#                same issue and on the same compound set. Source of the mouse
#                f(u,plasma) and f(u,brain) values below.
#                BOTH [Doran2005] and [Maurer2005] are closed access
#                (Unpaywall: is_oa false, no repository copy). Their values
#                are used here as quoted, with attribution, in Table 1 of
#                [Ma2024], which is open access and was read directly; that
#                second-hand status is recorded on each constant, and the
#                mouse numbers are NOT used to build anything, because this
#                module has no mouse PK for any of these drugs.
#   [Ma2024]     Ma Y, Jiang M, Javeria H, Tian D, Du Z. "Accurate prediction
#                of Kp,uu,brain based on experimental measurement of Kp,brain
#                and computed physicochemical properties of candidate
#                compounds in CNS drug discovery." Heliyon.
#                2024;10(2):e24304. doi:10.1016/j.heliyon.2024.e24304,
#                PMID 38298681, open access (PMC10828645). Its Table 1
#                ("Measured Kp,brain, fu,p, fu,b, Kp,uu,brain ... of 36
#                marketed published drugs in external validation group")
#                attributes every value used below to refs [1] = [Doran2005]
#                and [7] = [Maurer2005], i.e. mouse. The table is internally
#                consistent (Kp * fu,b / fu,p reproduces the printed Kp,uu to
#                within rounding for both drugs used here), which is the only
#                check available without the closed primaries.
#   [Friden2009] Fridén M, Winiwarter S, Jerndal G, Bengtsson O, Wan H,
#                Bredberg U, Hammarlund-Udenaes M, Antonsson M.
#                "Structure-brain exposure relationships in rat and human
#                using a novel data set of unbound drug concentrations in
#                brain interstitial and cerebrospinal fluids." J Med Chem.
#                2009;52(20):6233-43. doi:10.1021/jm901036q, PMID 19764786.
#                RAT. Closed access (Unpaywall: is_oa false). Its rat
#                diazepam Kp = 2.28 and Kp,uu = 1.07 are quoted, with
#                attribution to this paper, in Table 2 of Kim et al., Drug
#                Deliv. 2025 (PMC12667295, open access, read directly). Used
#                below ONLY as an independent corroboration of the derived
#                rat diazepam Kp,uu, never as the value itself.


# --- diazepam, rat: the one drug with a complete, species-matched set ------
#
# Every factor needed to turn a total plasma concentration into an unbound
# brain concentration exists for diazepam in the rat, and two of the three
# come from the same study and the same animals as this module's IP
# disposition parameters:
#
#     Cu,brain = f(u,brain) * Kp * C,plasma = 0.040 * 4.5 * C,plasma
#
# The Kp (4.5) and f(u,plasma) (0.196) are [Friedman1986]'s and are already
# defined above as DIAZEPAM_RAT_IP_BRAIN_PLASMA_RATIO and
# DIAZEPAM_RAT_IP_FREE_FRACTION; they are deliberately reused rather than
# restated so the two uses cannot drift apart.

#: Rat diazepam fraction unbound in WHOLE BRAIN homogenate, [Gustafsson2019]
#: Table 2, mean of N = 2 biological replicates (pools of 2-3 rats each).
DIAZEPAM_RAT_FU_BRAIN_WHOLE = 0.040
#: SD of the above, same table.
DIAZEPAM_RAT_FU_BRAIN_WHOLE_SD = 0.006
#: Regional rat diazepam f(u,brain), [Gustafsson2019] Table 2 means. Kept
#: because the paper's central finding is that brain tissue binding is NOT
#: homogeneous across regions, and Phase 3/4 make this codebase regional; the
#: whole-brain value above is what a non-regional model is entitled to use,
#: and the spread here (0.037-0.055, a factor of 1.5) is the error that
#: entitlement carries.
DIAZEPAM_RAT_FU_BRAIN_BY_REGION = {
    "frontal_cortex": 0.037,
    "parietal_cortex": 0.043,
    "basal_ganglia": 0.044,
    "cerebellum": 0.055,
    "hippocampus": 0.048,
    "whole_brain": 0.040,
}
#: [Friden2009]'s independently measured rat diazepam values, quoted
#: second-hand (see the source note). Corroboration only; nothing is built
#: from them. Note the Kp disagrees with [Friedman1986]'s 4.5 by a factor of
#: ~2 while the Kp,uu agrees with the value derived here to within 14%,
#: which is exactly the behavior the Kp/Kp,uu distinction predicts: Kp is
#: sensitive to tissue-binding assay and to the timepoint at which brain and
#: plasma are sampled, Kp,uu is not.
DIAZEPAM_RAT_KP_FRIDEN2009 = 2.28
DIAZEPAM_RAT_KP_UU_FRIDEN2009 = 1.07

#: Mouse diazepam, [Doran2005] (Kp) + [Maurer2005] (f(u,plasma),
#: f(u,brain)), quoted via [Ma2024] Table 1. Recorded, NOT used: this module
#: has no mouse diazepam PK (see SEARCHED_AND_EMPTY[("diazepam","mouse","*")]),
#: and a partition is not a licence to invent the kinetics it multiplies.
DIAZEPAM_MOUSE_KP = 2.00
DIAZEPAM_MOUSE_FU_PLASMA = 0.098
DIAZEPAM_MOUSE_FU_BRAIN = 0.050
DIAZEPAM_MOUSE_KP_UU_REPORTED = 1.02


# --- haloperidol: a rat Kp, and no rat unbound fractions -------------------

#: Rat haloperidol brain:plasma ratio, TOTAL/TOTAL, [Sunderland1987].
HALOPERIDOL_RAT_KP = 22.0
#: Mouse haloperidol, [Doran2005] + [Maurer2005] via [Ma2024]. Recorded, not
#: used: no mouse haloperidol PK exists here either.
HALOPERIDOL_MOUSE_KP = 13.0
HALOPERIDOL_MOUSE_FU_PLASMA = 0.087
HALOPERIDOL_MOUSE_FU_BRAIN = 0.007
HALOPERIDOL_MOUSE_KP_UU_REPORTED = 1.06


# --- fentanyl: a rat Kp read off a figure, and no rat unbound fractions ----

#: Rat fentanyl brain:plasma ratio (TOTAL brain ng/g : TOTAL plasma ng/mL),
#: by region, 1 h after 300 ug/kg SC, READ FROM Fig. 5B of [Canfield2025].
#: These are figure reads, not printed values; see the source note and
#: `FENTANYL_RAT_KP_FIGURE_READ_UNCERTAINTY`.
FENTANYL_RAT_KP_BY_REGION_FIGURE_READ = {
    "hippocampus": 15.0,
    "medulla": 18.0,
    "striatum": 16.0,
    "frontal_cortex": 17.0,
}
#: Absolute uncertainty on each of the above, in ratio units, from reading a
#: bar chart whose y-axis is gridded every 20 units. Deliberately generous.
FENTANYL_RAT_KP_FIGURE_READ_UNCERTAINTY = 2.0
#: Mean over the four regions read. Not a whole-brain measurement: it is the
#: average of four specific regions the paper chose for their relevance to
#: reward and respiration, and the true whole-brain value was not reported.
FENTANYL_RAT_KP_REGION_MEAN_FIGURE_READ = sum(
    FENTANYL_RAT_KP_BY_REGION_FIGURE_READ.values()
) / len(FENTANYL_RAT_KP_BY_REGION_FIGURE_READ)
#: Brain concentrations from the same figure's panel A, ng/g, used only for
#: the internal-consistency check below.
FENTANYL_RAT_BRAIN_NG_PER_G_BY_REGION_FIGURE_READ = {
    "hippocampus": 1100.0,
    "medulla": 1330.0,
    "striatum": 1060.0,
    "frontal_cortex": 1250.0,
}


def fentanyl_rat_brain_plasma_figure_read_cross_check():
    """Check the Fig. 5B ratio reads against the Fig. 5A concentration reads
    and [Canfield2025]'s own printed plasma parameters.

    Returns `(plasma_implied_by_figures_ng_per_mL,
    plasma_from_printed_pk_ng_per_mL, ratio)`.

    The two are derived completely differently: the first divides the mean
    figure-read brain concentration by the mean figure-read brain:plasma
    ratio, and the second propagates the paper's PRINTED Cmax (82.12 ng/mL at
    Tmax 0.5 h) forward to 1 h with its PRINTED terminal half-life (1.10 h).
    Agreement means the two panels were read off consistently and that the
    reads are compatible with the numbers the paper does print; it is not
    independent evidence that either read is exact.

    The second estimate is deliberately crude - a mono-exponential decline
    from Cmax ignores the absorption phase still contributing at 1 h - and it
    is biased low for that reason, so the check is one-sided: it verifies
    agreement to within a stated factor, not equality.
    """
    brain = sum(FENTANYL_RAT_BRAIN_NG_PER_G_BY_REGION_FIGURE_READ.values()) / len(
        FENTANYL_RAT_BRAIN_NG_PER_G_BY_REGION_FIGURE_READ
    )
    from_figures = brain / FENTANYL_RAT_KP_REGION_MEAN_FIGURE_READ
    elapsed_h = 1.0 - FENTANYL_RAT_SC_TMAX_H
    from_printed = FENTANYL_RAT_SC_CMAX_NG_PER_ML * math.exp(
        -LN2 * elapsed_h * 60.0 / FENTANYL_RAT_SC_TERMINAL_HALF_LIFE_MIN
    )
    return (from_figures, from_printed, from_figures / from_printed)


# --- searched-and-empty, partition edition ---------------------------------
#
# Same contract as SEARCHED_AND_EMPTY above: the record IS the error message
# a caller gets when they ask for something that does not exist.

SEARCHED_AND_EMPTY_BRAIN_PARTITION = {
    ("fentanyl", "rat", "fu_brain"): (
        "No measured rat brain unbound fraction f(u,brain) was found for "
        "fentanyl, so its rat brain:plasma ratio of ~%g [Canfield2025] "
        "cannot be converted into an unbound brain concentration and must "
        "not drive receptor occupancy. This is the whole Kp-vs-Kp,uu point: "
        "a ratio of ~16 for a lipophilic base is mostly brain tissue "
        "binding, and using it as if it were a Kp,uu would overstate free "
        "drug at the receptor by roughly 1/f(u,brain), i.e. one to two "
        "orders of magnitude.\n"
        "Searched (2026-08): PubMed (fentanyl AND brain AND plasma AND "
        "(partition OR ratio OR distribution) AND rat, 15 hits reviewed "
        "individually), Europe PMC open-access full text ('fentanyl AND "
        "\"brain-to-plasma\"', 159 hits; 'fentanyl AND \"fraction unbound\" "
        "AND brain AND rat', 20 hits), and Unpaywall on every closed "
        "candidate.\n"
        "Candidates found and why each is unusable: Yu C, Yuan M, Yang H, "
        "Zhuang X, Li H, Toxicol Sci 2018;164:353-62 (PMID 29669042) DID "
        "measure an unbound brain-to-plasma concentration ratio for fentanyl "
        "in rats at 40 ug/kg IV - its abstract says so explicitly, reporting "
        "that tariquidar raised it 2.9-fold - but the abstract gives only "
        "that fold change and no baseline value, and although Unpaywall "
        "marks the article open access the Oxford Academic PDF endpoint "
        "returns 403 to every automated request attempted (direct, "
        "browser-headered, and via a text-extraction proxy), so the value "
        "was never seen; quoting it second-hand is not possible either "
        "because no reachable paper restates it. Kalvass JC, Olson ER, "
        "Cassidy MP, Selley DE, Pollack GM, J Pharmacol Exp Ther "
        "2007;323:346-55 (PMID 17646430) reports unbound brain "
        "concentrations for fentanyl and six other opioids, but in MICE and "
        "behind a closed paywall, so it is wrong on species and unreadable "
        "besides. Bjorkman S, Stanski DR, Verotta D, Harashima H, "
        "Anesthesiology 1990;72:865-73 (PMID 2339802) measured steady-state "
        "tissue/BLOOD partition coefficients for fentanyl in 13 rat tissues "
        "including brain, which would be a better-controlled Kp than the "
        "figure read used here, but it is likewise 403-blocked at the "
        "publisher; a rat brain value of 4 appears in Table 1 of Front "
        "Pharmacol 2025;16 (PMC12575352) under the heading 'Kp of fentanyl "
        "in rats from report', with 13 tissue columns matching Bjorkman's 13 "
        "tissues, but that table carries no citation in its text, so the "
        "attribution is an inference and the number is not used. Note the "
        "unresolved tension it implies: 4 (steady-state infusion, "
        "tissue:blood) against ~16 (1 h post-SC, tissue:plasma), a 4-fold "
        "disagreement that only a readable primary can settle.\n"
        "What is NOT done about it: f(u,brain) is not borrowed from the "
        "mouse, not predicted from logP, and not set to 1. The rat fentanyl "
        "PK models therefore stay plasma-driven and say so "
        "(`model.receptor_compartment`)."
    ),
    ("haloperidol", "rat", "fu_brain"): (
        "No measured rat brain unbound fraction f(u,brain) and no measured "
        "rat plasma unbound fraction f(u,plasma) were found for haloperidol, "
        "so [Sunderland1987]'s rat brain:plasma ratio of 22 is a TOTAL Kp "
        "with no route to an unbound brain concentration and must not drive "
        "receptor occupancy.\n"
        "Searched (2026-08): PubMed (haloperidol AND brain AND plasma AND "
        "(ratio OR partition) AND rat, 20 hits; haloperidol rat brain plasma "
        "concentration ratio striatum, reviewed), Europe PMC open-access "
        "full text ('haloperidol AND \"brain to plasma ratio\"', 45 hits; "
        "'haloperidol AND \"Kpuu\"', 7 hits; 'haloperidol AND \"unbound "
        "brain-to-plasma\"', 7 hits), the Uppsala translational-PKPD "
        "literature specifically, and Unpaywall on every closed candidate.\n"
        "Candidates found and why each is unusable: Loryan I, Melander E, "
        "Svensson M, Payan M, König F, Jansson B, Hammarlund-Udenaes M, Mol "
        "Psychiatry 2016;21:1527-36 (PMID 26809840) is exactly the right "
        "study - it measures regional rat Kp,uu,brain AND regional rat "
        "f(u,brain) for haloperidol and four other antipsychotics in awake "
        "Sprague-Dawley rats - but it is closed access with no repository "
        "copy Unpaywall or DiVA could produce, and its abstract reports only "
        "the qualitative regional findings and fold-differences, no values; "
        "[Gustafsson2019], which IS open and is used here for diazepam, "
        "cites that paper for exactly these haloperidol numbers but does not "
        "reproduce them. Kapetanovic IM, Sweeney DJ, Rapoport SI, J "
        "Pharmacol Exp Ther 1982;221:434-8 (PMID 7077539) measured rat plasma "
        "and regional brain haloperidol by GC after 0.5 mg/kg i.p. and "
        "reports brain/plasma ratios, but only their variation with age is "
        "in the abstract and the full text is closed. Tsuneizumi T, Babb SM, "
        "Cohen BM, Biol Psychiatry 1992;32:817-24 (PMID 1360262) gives rat "
        "brain:serum ratios for nine neuroleptics by HPLC-EC after 5 days of "
        "1 mg/kg/day, which would be both chemically specific and a useful "
        "check on [Sunderland1987]'s radioreceptor assay, but its abstract "
        "reports only the across-drug range 0.18-62.5 and the full text is "
        "closed.\n"
        "Independent corroboration that the ORDER of the Kp is right, "
        "recorded but not used: the mouse Kp of 13.0 ([Doran2005] via "
        "[Ma2024]) is the same order as the rat 22, and the mouse f(u,brain) "
        "of 0.007 ([Maurer2005]) is what makes that mouse Kp,uu 1.06 rather "
        "than 13. Applying the mouse f(u,brain) to the rat Kp would give a "
        "plausible-looking rat Kp,uu of about 1.8, and it is not done: it "
        "would be a cross-species substitution in the one factor that "
        "carries all the difference between the two coefficients, and rule 2 "
        "of this codebase forbids exactly that."
    ),
    ("fentanyl", "mouse", "*"): (
        "No mouse brain partition of any kind is recorded for fentanyl. It "
        "would be unusable if it were: this module has no mouse fentanyl PK "
        "either (SEARCHED_AND_EMPTY[('fentanyl','mouse','*')]), so there "
        "would be no plasma concentration to multiply."
    ),
    ("*", "human", "*"): (
        "No human brain partition is recorded for any drug here, which is "
        "why the pre-P1.5 calibrated scenarios in scenarios.py remain "
        "plasma-driven and why P1.5 did not retro-fit them.\n"
        "Those scenarios (`mu_opioid_pk_models_calibrated`, "
        "`gaba_a_pk_models_calibrated`, `d2_pk_models_calibrated`) are built "
        "on HUMAN clinical PK - the fentanyl citrate label's 13 min "
        "redistribution half-life, naloxone's ~64 min serum half-life, "
        "[Mandelli1978]'s diazepam distribution phase, [Kudo1999]'s "
        "haloperidol terminal half-life - and dose in an achieved plasma "
        "concentration rather than in mg/kg. Attaching a RAT partition to "
        "human kinetics is the cross-species substitution rule 2 forbids, "
        "and no human Kp,uu was found for these drugs: human brain:plasma "
        "data for them is postmortem forensic material, where postmortem "
        "redistribution makes a brain:blood ratio a property of the interval "
        "since death as much as of the drug.\n"
        "Consequence, stated rather than hidden: those scenarios drive "
        "receptor occupancy with TOTAL PLASMA concentration, which is wrong "
        "in a known direction (it ignores both plasma protein binding, which "
        "lowers the driving concentration, and brain partition, which for "
        "these lipophilic bases raises it). Their PK models report "
        "`receptor_compartment == 'total plasma (no brain partition "
        "attached)'`, and P1.5 changed none of their numbers. Phase 2 moves "
        "them onto the rat dose bridge, which is where a species-matched "
        "partition becomes available."
    ),
}


# --- constructors ----------------------------------------------------------


def diazepam_rat_brain_partition(region="whole_brain"):
    """Rat diazepam plasma -> brain partition, on the TOTAL Kp basis with the
    measured unbound fractions attached, so it yields an unbound brain
    concentration and can drive the receptor layer.

    Kp = 4.5 and f(u,plasma) = 0.196 are [Friedman1986]'s, from the same
    5 mg/kg IP study whose disposition this module's `diazepam_rat_ip_pk`
    already uses. f(u,brain) is [Gustafsson2019]'s, from a different
    laboratory and a different (equilibrium-dialysis) experiment, so the
    resulting Kp,uu is a two-study composite in the same sense as the IP
    disposition model, and is labeled as one.

    Kp,uu = Kp * f(u,brain) / f(u,plasma) = 4.5 * 0.040 / 0.196 = 0.918.
    Independent corroboration: [Friden2009] measured 1.07 in the rat. The
    14% gap is small next to the 2-fold gap between the two studies' Kp
    values (4.5 vs 2.28), which is the expected behavior - see
    `DIAZEPAM_RAT_KP_FRIDEN2009`.

    Both values sit near 1, i.e. diazepam is at passive equilibrium across
    the rat blood-brain barrier with no net efflux, which is the independent
    qualitative expectation for a small lipophilic neutral benzodiazepine
    that is not a P-glycoprotein substrate.

    `region` selects which [Gustafsson2019] f(u,brain) to use. The default,
    whole brain, is what a non-regional model is entitled to; the regional
    values exist because that paper's finding is that tissue binding is not
    homogeneous, and Phases 3-4 make this codebase regional.
    """
    if region not in DIAZEPAM_RAT_FU_BRAIN_BY_REGION:
        raise KeyError(
            "no [Gustafsson2019] rat diazepam f(u,brain) for region %r; "
            "available: %s"
            % (region, ", ".join(sorted(DIAZEPAM_RAT_FU_BRAIN_BY_REGION)))
        )
    return BrainPartition(
        drug="diazepam",
        species="rat",
        basis="Kp",
        ratio=DIAZEPAM_RAT_IP_BRAIN_PLASMA_RATIO,
        fu_plasma=DIAZEPAM_RAT_IP_FREE_FRACTION,
        fu_brain=DIAZEPAM_RAT_FU_BRAIN_BY_REGION[region],
        region=region,
        source=(
            "Kp 4.5 +/- 0.1 and f(u,plasma) 0.196 from [Friedman1986] "
            "(rat, 5 mg/kg IP); f(u,brain) %g from [Gustafsson2019] Table 2 "
            "(Sprague-Dawley rat, %s, brain-homogenate equilibrium dialysis)"
            % (DIAZEPAM_RAT_FU_BRAIN_BY_REGION[region], region)
        ),
        equilibration_evidence=(
            "[Friedman1986] measured a brain half-life of 0.89 h against a "
            "plasma half-life of 0.88 h in the same animals: brain and "
            "plasma decline in parallel, so treating brain as instantaneously "
            "equilibrated with plasma is supported rather than assumed on "
            "this model's minute-to-hour timescale."
        ),
        note=(
            "Two-study composite. [Friedman1986]'s own f(u,plasma) of 0.196 "
            "disagrees with [Klotz1976]'s 0.137 in the rat (a 43% relative "
            "spread between two equilibrium-dialysis assays; see "
            "DIAZEPAM_RAT_IP_FREE_FRACTION). Using [Klotz1976]'s instead "
            "would raise Kp,uu from 0.918 to 1.31 and would NOT change the "
            "unbound brain factor at all, because that factor is "
            "f(u,brain)*Kp and does not contain f(u,plasma) - which is worth "
            "stating, since it means the receptor-driving concentration is "
            "insensitive to the one number the two labs disagree about."
        ),
    )


def diazepam_mouse_brain_partition():
    """Mouse diazepam partition. Complete on paper, and unusable in practice:
    this module has no mouse diazepam PK, so there is no plasma concentration
    for it to multiply. Recorded so the roadmap's eventual move to mouse
    (Phases 3-4) starts from a sourced number rather than a search.

    Kp * f(u,brain) / f(u,plasma) = 2.00 * 0.050 / 0.098 = 1.020, against the
    1.02 [Ma2024]'s table prints - the internal-consistency check that is the
    only verification available while [Doran2005] and [Maurer2005] are
    unreadable.
    """
    return BrainPartition(
        drug="diazepam",
        species="mouse",
        basis="Kp",
        ratio=DIAZEPAM_MOUSE_KP,
        fu_plasma=DIAZEPAM_MOUSE_FU_PLASMA,
        fu_brain=DIAZEPAM_MOUSE_FU_BRAIN,
        source=(
            "Kp from [Doran2005], f(u,plasma)/f(u,brain) from [Maurer2005]; "
            "both quoted second-hand via [Ma2024] Table 1 (open access, read "
            "directly), the primaries being closed"
        ),
        note=(
            "MOUSE. Not attached to anything: no mouse diazepam PK exists in "
            "this module. Second-hand values."
        ),
    )


def haloperidol_rat_brain_partition():
    """Rat haloperidol partition: a TOTAL brain:plasma Kp of 22
    [Sunderland1987], and nothing else.

    This object exists and is cited, and it deliberately CANNOT drive
    receptor occupancy: `can_drive_receptor` is False and
    `unbound_brain_factor` raises, because no rat f(u,brain) for haloperidol
    was found (see
    SEARCHED_AND_EMPTY_BRAIN_PARTITION[("haloperidol","rat","fu_brain")]).
    What it can do is report a total brain concentration, which is the
    quantity a tissue-homogenate measurement would return and therefore the
    right thing to compare a published brain concentration against.

    The Kp itself carries a real assay caveat: [Sunderland1987] used a
    radioreceptor assay, which measures neuroleptic-equivalent D2-binding
    activity rather than chemically specific haloperidol.
    """
    return BrainPartition(
        drug="haloperidol",
        species="rat",
        basis="Kp",
        ratio=HALOPERIDOL_RAT_KP,
        source="[Sunderland1987], rat, radioreceptor assay",
        note=(
            "TOTAL brain:plasma only. Radioreceptor assay, so active "
            "metabolites contribute to both numerator and denominator; dose, "
            "route, timing, strain, sex and n are not obtainable because the "
            "full text is closed. No rat f(u,brain) found, so this cannot "
            "give an unbound brain concentration. The mouse Kp of 13.0 "
            "([Doran2005] via [Ma2024]) is recorded separately as an "
            "order-of-magnitude corroboration only."
        ),
    )


def haloperidol_mouse_brain_partition():
    """Mouse haloperidol partition. Complete, second-hand, and unattached, for
    the same reason as `diazepam_mouse_brain_partition`: no mouse haloperidol
    PK exists here.

    Kp * f(u,brain) / f(u,plasma) = 13.0 * 0.007 / 0.087 = 1.046 against the
    printed 1.06. Worth reading next to the rat entry: the same drug shows a
    Kp of 13-22 and a Kp,uu of ~1, i.e. it is at passive equilibrium across
    the barrier and merely very heavily bound to brain tissue. Driving a
    receptor with the Kp would be wrong by the factor 1/f(u,brain) ~ 140.
    """
    return BrainPartition(
        drug="haloperidol",
        species="mouse",
        basis="Kp",
        ratio=HALOPERIDOL_MOUSE_KP,
        fu_plasma=HALOPERIDOL_MOUSE_FU_PLASMA,
        fu_brain=HALOPERIDOL_MOUSE_FU_BRAIN,
        source=(
            "Kp from [Doran2005], f(u,plasma)/f(u,brain) from [Maurer2005]; "
            "both quoted second-hand via [Ma2024] Table 1"
        ),
        note=(
            "MOUSE. Not attached to anything: no mouse haloperidol PK exists "
            "in this module. Second-hand values."
        ),
    )


def fentanyl_rat_brain_partition(region="region_mean"):
    """Rat fentanyl partition: a TOTAL brain:plasma Kp of about 15-18
    depending on region, READ OFF Fig. 5B of [Canfield2025].

    Like the haloperidol one, this deliberately cannot drive receptor
    occupancy: no rat f(u,brain) for fentanyl was found (see
    SEARCHED_AND_EMPTY_BRAIN_PARTITION[("fentanyl","rat","fu_brain")]).

    `region` is one of the four the paper resolved, or "region_mean" for the
    mean of the four - which is NOT a whole-brain measurement, only an
    average of the regions that paper chose.
    """
    if region == "region_mean":
        ratio = FENTANYL_RAT_KP_REGION_MEAN_FIGURE_READ
    elif region in FENTANYL_RAT_KP_BY_REGION_FIGURE_READ:
        ratio = FENTANYL_RAT_KP_BY_REGION_FIGURE_READ[region]
    else:
        raise KeyError(
            "no [Canfield2025] rat fentanyl brain:plasma read for region %r; "
            "available: region_mean, %s"
            % (region, ", ".join(sorted(FENTANYL_RAT_KP_BY_REGION_FIGURE_READ)))
        )
    return BrainPartition(
        drug="fentanyl",
        species="rat",
        basis="Kp",
        ratio=ratio,
        region=region,
        source=(
            "[Canfield2025] Fig. 5B, FIGURE READ (+/- %g ratio units), six "
            "male Sprague-Dawley rats, 300 ug/kg SC, 1 h post-dose"
            % FENTANYL_RAT_KP_FIGURE_READ_UNCERTAINTY
        ),
        equilibration_evidence=(
            "[HugMurphy1981]: after 50 ug/kg IV in rats, 'fentanyl "
            "concentrations in brain, heart, and lung equilibrated with that "
            "in plasma before 1.5 min and declined at the same rate'. Fast "
            "relative to this codebase's minute-scale time unit, so an "
            "instantaneous partition is supported for fentanyl too - which "
            "matters because it means the missing piece here is only "
            "f(u,brain), not the model structure."
        ),
        note=(
            "TOTAL brain:plasma only, and a figure read rather than a "
            "printed value. A steady-state rat tissue:blood value of 4 for "
            "brain may exist in [Bjorkman1990]; it is unread and 4-fold "
            "lower, and that disagreement is unresolved. No rat f(u,brain) "
            "found, so this cannot give an unbound brain concentration."
        ),
    )


def brain_partition_for(drug, species, require_receptor_drivable=True):
    """The sourced partition for `drug`/`species`, or None.

    With `require_receptor_drivable=True` (the default, and what the PK
    builders use), a partition that exists but cannot yield an unbound brain
    concentration returns None rather than being attached, because attaching
    it would make the PK constructor raise and take a route that DOES have
    sourced kinetics offline. The result is not silent: the model then
    reports `receptor_compartment == 'total plasma (no brain partition
    attached)'`, and `brain_partition_status` says exactly which of the three
    cases applies and why.
    """
    builders = {
        ("diazepam", "rat"): diazepam_rat_brain_partition,
        ("diazepam", "mouse"): diazepam_mouse_brain_partition,
        ("haloperidol", "rat"): haloperidol_rat_brain_partition,
        ("haloperidol", "mouse"): haloperidol_mouse_brain_partition,
        ("fentanyl", "rat"): fentanyl_rat_brain_partition,
    }
    builder = builders.get((str(drug).lower(), str(species).lower()))
    if builder is None:
        return None
    partition = builder()
    if require_receptor_drivable and not partition.can_drive_receptor:
        return None
    return partition


def brain_partition_status(drug, species):
    """`(status, explanation)` for one drug/species, where status is one of
    "receptor-drivable", "total-brain-only" or "absent".

    Exists so that "this simulation ran on plasma" is a queryable fact with a
    reason attached, rather than something a reader has to infer from the
    absence of an attribute.
    """
    key = (str(drug).lower(), str(species).lower())
    partition = brain_partition_for(*key, require_receptor_drivable=False)
    if partition is None:
        record = SEARCHED_AND_EMPTY_BRAIN_PARTITION.get(
            key + ("*",)
        ) or SEARCHED_AND_EMPTY_BRAIN_PARTITION.get(("*", key[1], "*"))
        return (
            "absent",
            record
            or "No brain partition has been sourced for %s in the %s, and no "
            "search for one has been recorded." % key,
        )
    if partition.can_drive_receptor:
        return ("receptor-drivable", partition.source)
    return (
        "total-brain-only",
        SEARCHED_AND_EMPTY_BRAIN_PARTITION.get(
            key + ("fu_brain",), partition.note or ""
        ),
    )


__all__ = [
    "UnsourcedParameterError",
    "SEARCHED_AND_EMPTY",
    "SEARCHED_AND_EMPTY_BRAIN_PARTITION",
    "USE_SOURCED_BRAIN_PARTITION",
    "brain_partition_for",
    "brain_partition_status",
    "DIAZEPAM_RAT_FU_BRAIN_WHOLE",
    "DIAZEPAM_RAT_FU_BRAIN_WHOLE_SD",
    "DIAZEPAM_RAT_FU_BRAIN_BY_REGION",
    "DIAZEPAM_RAT_KP_FRIDEN2009",
    "DIAZEPAM_RAT_KP_UU_FRIDEN2009",
    "DIAZEPAM_MOUSE_KP",
    "DIAZEPAM_MOUSE_FU_PLASMA",
    "DIAZEPAM_MOUSE_FU_BRAIN",
    "DIAZEPAM_MOUSE_KP_UU_REPORTED",
    "diazepam_rat_brain_partition",
    "diazepam_mouse_brain_partition",
    "HALOPERIDOL_RAT_KP",
    "HALOPERIDOL_MOUSE_KP",
    "HALOPERIDOL_MOUSE_FU_PLASMA",
    "HALOPERIDOL_MOUSE_FU_BRAIN",
    "HALOPERIDOL_MOUSE_KP_UU_REPORTED",
    "haloperidol_rat_brain_partition",
    "haloperidol_mouse_brain_partition",
    "FENTANYL_RAT_KP_BY_REGION_FIGURE_READ",
    "FENTANYL_RAT_KP_FIGURE_READ_UNCERTAINTY",
    "FENTANYL_RAT_KP_REGION_MEAN_FIGURE_READ",
    "FENTANYL_RAT_BRAIN_NG_PER_G_BY_REGION_FIGURE_READ",
    "fentanyl_rat_brain_partition",
    "fentanyl_rat_brain_plasma_figure_read_cross_check",
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
    "DIAZEPAM_MOLAR_MASS_G_PER_MOL",
    "NORDIAZEPAM_MOLAR_MASS_G_PER_MOL",
    "DIAZEPAM_RAT_IV_CL_ML_PER_MIN_PER_KG",
    "DIAZEPAM_RAT_IV_TERMINAL_HALF_LIFE_MIN",
    "DIAZEPAM_RAT_IV_TERMINAL_HALF_LIFE_INTERVAL_MIN",
    "DIAZEPAM_ALPHA_HALF_LIFE_RANGE_ALL_SPECIES_H",
    "DIAZEPAM_RAT_PLASMA_BOUND_FRACTION",
    "NORDIAZEPAM_RAT_PLASMA_BOUND_FRACTION",
    "DIAZEPAM_RAT_IV_F",
    "DIAZEPAM_RAT_IV_BETA_PER_MIN",
    "DIAZEPAM_RAT_IV_VBETA_L_PER_KG",
    "DIAZEPAM_RAT_IP_VD_OVER_F_L_PER_KG",
    "DIAZEPAM_RAT_IP_CL_OVER_F_ML_PER_MIN_PER_KG",
    "DIAZEPAM_RAT_IP_TERMINAL_HALF_LIFE_MIN",
    "DIAZEPAM_RAT_IP_BRAIN_HALF_LIFE_MIN",
    "DIAZEPAM_RAT_IP_BRAIN_PLASMA_RATIO",
    "DIAZEPAM_RAT_IP_FREE_FRACTION",
    "DIAZEPAM_RAT_IP_SOURCE_DOSE_MG_PER_KG",
    "DIAZEPAM_RAT_IP_SOURCE_N_PER_TIMEPOINT",
    "DIAZEPAM_RAT_IP_SOURCE_SAMPLING_TIMES_MIN",
    "DIAZEPAM_RAT_IP_KA_PER_MIN",
    "DIAZEPAM_RAT_IP_KA_SOURCE_K12_PER_MIN",
    "DIAZEPAM_RAT_IP_KA_SOURCE_K21_PER_MIN",
    "DIAZEPAM_RAT_IP_KA_SOURCE_KEL_PER_MIN",
    "DIAZEPAM_RAT_IP_KA_SOURCE_AGE_DAYS",
    "DIAZEPAM_RAT_IP_KA_SOURCE_DOSE_MG_PER_KG",
    "DIAZEPAM_RAT_PO_CMAX_UG_PER_L",
    "DIAZEPAM_RAT_PO_CMAX_SD_UG_PER_L",
    "DIAZEPAM_RAT_PO_TMAX_H",
    "DIAZEPAM_RAT_PO_TMAX_SD_H",
    "DIAZEPAM_RAT_PO_TERMINAL_HALF_LIFE_H",
    "DIAZEPAM_RAT_PO_TERMINAL_HALF_LIFE_SD_H",
    "DIAZEPAM_RAT_PO_AUC_0_T_UG_H_PER_L",
    "DIAZEPAM_RAT_PO_AUC_0_INF_UG_H_PER_L",
    "DIAZEPAM_RAT_PO_CL_OVER_F_L_PER_H_PER_KG",
    "DIAZEPAM_RAT_PO_VZ_OVER_F_L_PER_KG",
    "DIAZEPAM_RAT_PO_VZ_OVER_F_SD_L_PER_KG",
    "DIAZEPAM_RAT_PO_MRT_0_INF_H",
    "DIAZEPAM_RAT_PO_SOURCE_DOSE_MG_PER_KG",
    "DIAZEPAM_RAT_PO_SOURCE_BODY_WEIGHT_KG",
    "DIAZEPAM_RAT_PO_SOURCE_N",
    "DIAZEPAM_RAT_PO_SOURCE_SAMPLING_TIMES_H",
    "NORDIAZEPAM_RAT_PO_CMAX_UG_PER_L",
    "NORDIAZEPAM_RAT_PO_TMAX_H",
    "NORDIAZEPAM_RAT_PO_TERMINAL_HALF_LIFE_H",
    "NORDIAZEPAM_RAT_PO_AUC_0_T_UG_H_PER_L",
    "NORDIAZEPAM_RAT_PO_AUC_0_INF_UG_H_PER_L",
    "DIAZEPAM_ACTIVE_METABOLITE_LIMITATION",
    "diazepam_rat_pk",
    "diazepam_mouse_pk",
    "diazepam_rat_iv_pk",
    "diazepam_rat_ip_pk",
    "diazepam_rat_iv_terminal_half_life_min",
    "diazepam_rat_ip_terminal_half_life_min",
    "diazepam_rat_ip_reported_half_life_consistency",
    "diazepam_rat_ip_composite_disposition_cross_check",
    "diazepam_rat_ip_predicted_tmax_min",
    "diazepam_rat_ip_bioavailability",
    "diazepam_rat_po_apparent_volume_l_per_kg",
    "diazepam_rat_po_clearance_auc_consistency",
    "diazepam_rat_po_bioavailability",
    "diazepam_rat_po_ka_is_not_identified",
    "diazepam_hepatic_extraction_note",
    "diazepam_nordiazepam_molar_exposure_ratio",
]
