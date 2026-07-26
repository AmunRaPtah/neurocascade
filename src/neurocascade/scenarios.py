"""Illustrative parameter presets for worked examples.

IMPORTANT: the kon/koff/ke/tau/gain/weight values below are hand-set to
produce qualitatively and mechanistically correct dynamics (a mu-opioid
agonist disinhibits VTA dopamine neurons; an antagonist blocks and can
displace an already-bound agonist). They are NOT fit to or sourced from any
specific receptor-binding or pharmacokinetic dataset: treat them as
placeholders awaiting calibration against real Ki/kon/koff and PK
literature, the same way the CCT model's priors were literature-elicited
before use. Do not cite these numbers as measured values.
"""

from .behavior import BehaviorReadout
from .circuit import CircuitNode
from .network import CircuitNetwork
from .pk import OneCompartmentPK
from .receptor import Ligand, ReceptorSite


def mu_opioid_ligands():
    fentanyl = Ligand(name="fentanyl", kon=1.0, koff=0.15, efficacy=1.0)
    naloxone = Ligand(name="naloxone", kon=1.2, koff=0.05, efficacy=0.0)
    return [fentanyl, naloxone]


def mu_opioid_receptor_site():
    return ReceptorSite(name="mu_opioid_on_VTA_GABA", ligands=mu_opioid_ligands())


def mu_opioid_pk_models():
    # Relative ordering (naloxone clears faster than fentanyl) is meant to
    # be directionally right; absolute values are illustrative only.
    return [OneCompartmentPK(ke=0.05), OneCompartmentPK(ke=0.20)]


def reward_circuit_network():
    """VTA -> NAc -> PFC reward circuit, textbook opioid-disinhibition wiring:
    mu-agonist suppresses VTA GABAergic interneurons, which tonically inhibit
    VTA dopamine neurons; suppressing the brake increases DA firing, which
    drives NAc, which drives PFC; PFC exerts top-down feedback control.
    """
    nodes = [
        CircuitNode(name="VTA_GABA", tau=1.0, baseline_input=0.5, gain=4.0),
        CircuitNode(name="VTA_DA", tau=1.5, baseline_input=0.2, gain=4.0),
        CircuitNode(name="NAc", tau=2.0, baseline_input=0.0, gain=4.0),
        CircuitNode(name="PFC", tau=3.0, baseline_input=0.3, gain=4.0),
    ]
    idx = {node.name: i for i, node in enumerate(nodes)}
    n = len(nodes)
    W = [[0.0] * n for _ in range(n)]
    W[idx["VTA_DA"]][idx["VTA_GABA"]] = -2.5  # local GABA inhibits DA neurons
    W[idx["NAc"]][idx["VTA_DA"]] = 2.0  # mesolimbic dopamine projection
    W[idx["PFC"]][idx["NAc"]] = 1.5  # NAc drives PFC
    W[idx["VTA_DA"]][idx["PFC"]] = -0.8  # top-down feedback control
    W[idx["VTA_GABA"]][idx["PFC"]] = 0.6  # top-down recruits local inhibition
    return CircuitNetwork(nodes=nodes, connectivity=W)


def mu_opioid_coupling():
    # Mu receptors are Gi-coupled and sit on VTA GABAergic interneurons:
    # agonist activation SUPPRESSES that node's drive (disinhibition).
    return {"VTA_GABA": -3.0}


def incentive_salience_readout():
    return BehaviorReadout(
        name="incentive_salience", weights={"NAc": 0.7, "VTA_DA": 0.3}
    )


# ---------------------------------------------------------------------------
# Calibrated mu-opioid receptor + fentanyl/naloxone PK, grounded in published
# data (Track A). Units: concentration in nM, time in minutes, kon in
# nM^-1 min^-1, koff/ke in min^-1. Everything below this line is a real-data
# parameter set, kept deliberately separate from the illustrative functions
# above so the two are never confused. The reward circuit (`reward_circuit_
# network`, `mu_opioid_coupling`, `incentive_salience_readout`) is reused
# as-is: this pass only calibrates receptor binding and PK, not circuit
# connectivity/gain/tau, which have no literature target and remain
# illustrative (see docs/architecture.md).
#
# Sources (see inline citations per parameter below):
#   [Volpe2011]  Volpe DA, McMahon Tobin GA, Mellon RD, Katki AG, Parker RJ,
#                Colatsky T, Kropp TJ, Verbois SL. "Uniform assessment and
#                ranking of opioid mu receptor binding constants for selected
#                opioid drugs." Regul Toxicol Pharmacol. 2011;59(3):385-90.
#                doi:10.1016/j.yrtph.2010.12.007. Single standardized
#                [3H]-DAMGO competition-binding assay on a recombinant human
#                MOR membrane preparation; fentanyl and naloxone were run in
#                the SAME assay, so their Ki's are directly comparable
#                (fentanyl Ki = 1.346 nM; naloxone, used as the assay's
#                reference ligand, Ki = 1.518 +/- 0.065 nM, IC50 = 5.926 +/-
#                0.253 nM, both figures read directly from the paper text).
#   [Mann2022]   Mann J, Samieegohar M, Chaturbedi A, Zirkle J, Han X,
#                Ahmadi SF, Eshleman A, Janowsky A, Wolfrum K, Swanson T,
#                Bloom S, Dahan A, Olofsen E, Florian J, Strauss DG, Li Z.
#                "Development of a Translational Model to Assess the Impact
#                of Opioid Overdose and Naloxone Dosing on Respiratory
#                Depression and Cardiac Arrest." Clin Pharmacol Ther.
#                2022;112(5):1020-1032. doi:10.1002/cpt.2696. FDA/CDER
#                mechanistic PK-PD model of naloxone reversal; mu-receptor
#                association/dissociation rate constants fit per-ligand from
#                in vitro binding-kinetics assays. Code + fit parameters are
#                public: github.com/FDA/Mechanistic-PK-PD-Model-to-Rescue-
#                Opioid-Overdose, Ligand_Data/{Fentanyl_A,NALOXONE}_
#                parameters/*_pars.txt: fentanyl koff = 0.004331 s^-1
#                (t1/2(diss) ~ 160 s, i.e. ~2.7 min, independently
#                corroborated as "~4 min" by a computational residence-time
#                study, Mahinthichaichan et al. 2021 JACS Au
#                doi:10.1021/jacsau.1c00341); naloxone koff = 0.03959 s^-1
#                (t1/2(diss) ~ 17.5 s). NOTE: Mann et al.'s Kon values use a
#                nonstandard Hill-exponent parametrization (units
#                pM^-n * s^-1, n ~ 0.84-0.86) that isn't dimensionally
#                compatible with this codebase's strictly bimolecular (n=1)
#                occupancy ODE, so we do not reuse their Kon directly; see
#                the kon derivation below instead.
#   [FDA-label]  Fentanyl Citrate Injection, USP prescribing information
#                (accessdata.fda.gov/drugsatfda_docs/label/.../016619...
#                lbl.pdf; figures corroborated across multiple label
#                revisions 2016-2025). Three-compartment IV PK: distribution
#                t1/2 = 1.7 min, redistribution t1/2 = 13 min, terminal
#                elimination t1/2 = 219 min, Vd = 4 L/kg.
#   [Naloxone-PK] Handal KA, Schauben JL, Salamone FR. "Naloxone." Ann Emerg
#                Med. 1983;9(4):438-45 (dosing-nomogram pharmacokinetic
#                analysis; biexponential decline). Corroborated by a more
#                recent review, Rzasa Lynn R, Galinkin JL. "Clinical
#                Pharmacokinetics and Pharmacodynamics of Naloxone." Clin
#                Pharmacokinet. 2024. IV serum half-life ~30-81 min, mean
#                ~64 min.
#   [Moss2020]   Moss RB, Pryor MM, Baillie R, Kudrycki K, Friedrich C,
#                Reed M, Carlo DJ. "Higher naloxone dosing in a quantitative
#                systems pharmacology model that predicts naloxone-fentanyl
#                competition at the opioid mu receptor level." PLoS ONE.
#                2020;15(6):e0234683. doi:10.1371/journal.pone.0234683
#                (open access, PMC7297366). Independent QSP model using
#                fentanyl Ki = 1.35 nM and naloxone Ki = 1.115 nM (close to
#                the [Volpe2011] values above, corroborating that ~1-1.5 nM
#                range) and reporting baseline (no naloxone) mu-receptor
#                occupancy of 73%/97%/98.9% at fentanyl exposure levels of
#                25/50/75 ng/mL, used below as a validation target, with
#                caveats (see test_scenarios_calibrated.py): that paper's
#                occupancy is computed from an internal brain/effect-site
#                concentration produced by its own multi-compartment brain
#                PK submodel, not the plasma-equivalent single concentration
#                variable this codebase tracks, so we validate the
#                direction/order of magnitude (near-saturating occupancy at
#                overdose-range exposure), not an exact percentage match.
#
# Derivation of kon: none of the sources above report a kon for fentanyl or
# naloxone in units compatible with this codebase's ODE
# (d(occ)/dt = kon*C*(1-sum(occ)) - koff*occ, which requires kon in
# nM^-1 min^-1). We derive it from the two numbers that ARE independently
# measured and dimensionally simple: Ki (a true equilibrium constant) and
# koff (a clean first-order rate constant), via the standard reversible
# bimolecular relationship Kd = koff / kon, i.e. kon = koff / Kd, taking
# Kd ~= Ki (the usual approximation for a reversible orthosteric ligand,
# same one implicit in the Cheng-Prusoff equation Volpe et al. used to
# compute Ki from IC50).
def mu_opioid_ligands_calibrated():
    fentanyl = Ligand(
        name="fentanyl",
        kon=0.1931,   # nM^-1 min^-1; koff/Ki = (0.004331 s^-1 * 60) / 1.346 nM
        koff=0.2599,  # min^-1; 0.004331 s^-1 [Mann2022]
        efficacy=1.0,
    )
    naloxone = Ligand(
        name="naloxone",
        kon=1.5648,   # nM^-1 min^-1; koff/Ki = (0.03959 s^-1 * 60) / 1.518 nM
        koff=2.3754,  # min^-1; 0.03959 s^-1 [Mann2022]
        efficacy=0.0,
    )
    return [fentanyl, naloxone]


def mu_opioid_receptor_site_calibrated():
    return ReceptorSite(name="mu_opioid_on_VTA_GABA", ligands=mu_opioid_ligands_calibrated())


def mu_opioid_pk_models_calibrated():
    # One-compartment collapse of each drug's real multi-compartment IV PK,
    # chosen to match this scenario's acute-bolus/reversal timescale
    # (minutes to ~1 h) rather than steady-state/terminal disposition. This
    # is an explicit, cited simplification, not a claim that either drug
    # truly follows one-compartment kinetics.
    #   fentanyl ke: ln(2) / 13 min, the REDISTRIBUTION-phase half-life
    #     [FDA-label], the dominant decline in the first ~1 h after a
    #     single IV bolus (the reason fentanyl's clinical duration of action
    #     is short despite a long, 219 min, terminal half-life driven by
    #     slow release from fat/muscle depots). Using the terminal phase
    #     instead would over-predict how long a single bolus stays on
    #     board within this scenario's time window.
    #   naloxone ke: ln(2) / 64 min, the mean reported IV serum half-life
    #     [Naloxone-PK].
    # Net effect: in this one-compartment approximation, fentanyl's
    # concentration falls FASTER than naloxone's (ke 0.0533 vs 0.0108
    # min^-1), the reverse of the illustrative placeholder's assumed
    # ordering. This is a real and clinically important asymmetry: it is
    # naloxone's short RECEPTOR residence time (see koff above, ~17.5 s vs
    # fentanyl's ~160 s) and short duration of antagonism, not a faster
    # plasma clearance, that drives naloxone reversal to wear off before a
    # large/long-acting fentanyl exposure does (renarcotization risk), a
    # phenomenon that requires fentanyl's slow terminal elimination phase
    # to manifest, which this one-compartment collapse cannot represent.
    return [OneCompartmentPK(ke=0.05332), OneCompartmentPK(ke=0.01083)]
