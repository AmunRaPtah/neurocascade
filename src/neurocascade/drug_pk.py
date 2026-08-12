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

Fentanyl (P1.2) and diazepam (P1.4) are not here yet.
"""

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
]
