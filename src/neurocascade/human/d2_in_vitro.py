"""Human in-vitro D2 measurements for haloperidol: affinity and intrinsic efficacy (P19.4).

The headline result of the search is that **there is no single human in-vitro D2
Ki for haloperidol**. Seven primary measurements at cloned human D2 were
retrieved in full text, and they span 0.13 to 8.85 nM, a factor of 68. None of
them is wrong. They differ because a displacement Ki is a property of the assay
as much as of the ligand, and these assays differ in radioligand, tracer class,
receptor isoform, sodium concentration, guanine nucleotide and temperature.

That is the effect Seeman & Van Tol 1995 reported across three radioligands,
reproduced here from sources that could be retrieved. Seeman & Van Tol itself is
paywalled and contributes no number to this module.

Two patterns are worth stating as observations rather than as mechanism:

* The spread tracks **tracer class and guanine-nucleotide state** more closely
  than sodium alone. The [3H]spiperone filtration assays sit at 2.9 to 8.85 nM;
  the agonist-tracer and [3H]-N-methylspiperone assays sit at 0.13 to 0.46 nM.
  Sodium does not carry it by itself, since the highest sodium of any source
  (Sykes 2017, 138 mM) reports one of the lowest values. Two of the
  [3H]spiperone sources never print their buffer, so the ion effect cannot be
  isolated from what is printed.
* **Binding Ki and functional Kb diverge by 20 to 60-fold inside single papers.**
  Kaczor 2023a reports a binding Ki of 6.56 nM and its own cAMP Kb of 0.11 nM;
  Kaczor 2023b reports 7.41 nM against 0.29 nM. Same lab, same receptor, same
  paper. Any ratio taken against an in-vivo EC50 therefore lands somewhere
  different depending on whether the in-vitro denominator is a binding or a
  functional constant. Both kinds are recorded, kept apart.

No primary human **tissue** measurement was found. Searches over human
post-mortem brain, caudate, striatum and pituitary returned only reviews and
conference-abstract collections. Every retrievable primary human D2 haloperidol
affinity is from a heterologous cell line, CHO-K1 or HEK293T. That is recorded as
a result of the search rather than as a gap in it.

On intrinsic efficacy, the search was not empty. Haloperidol is an **inverse
agonist** at human D2, not a neutral antagonist, by two independent functional
readouts. See :data:`HALOPERIDOL_HUMAN_D2_INTRINSIC_EFFICACY` and the sign
convention documented on :data:`EFFICACY_SCALE`.
"""
from dataclasses import dataclass

_RETRIEVAL_DATE = "2026-09-03"

#: The sign convention for the `efficacy` field on a receptor parameter record.
#:
#: The scale is **asymmetric and the two ends have different references**, which
#: is easy to misread as a symmetric -1 to +1 axis:
#:
#: * ``0.0`` is a neutral antagonist, one that occupies without shifting the
#:   agonist-independent signal in either direction.
#: * Positive values are agonism, referenced to the endogenous agonist.
#: * Negative values are inverse agonism, referenced to **the maximal inverse
#:   response of the reference compound in that assay**, which for both human D2
#:   sources here is (+)-butaclamol, not to the endogenous agonist.
#:
#: So an efficacy of -0.95 means 95% of (+)-butaclamol's maximal inverse-agonist
#: effect in that system. It does not mean 95% as far below basal as dopamine is
#: above it. Nothing in this repository converts between the two ends of the
#: scale, and this note exists so that nothing later does so by accident.
EFFICACY_SCALE = (
    "0 = neutral antagonist; positive = agonism referenced to the endogenous "
    "agonist; negative = inverse agonism referenced to the reference compound's "
    "maximal inverse response in that assay, here (+)-butaclamol. The two ends "
    "do not share a reference and are not interconvertible."
)


@dataclass(frozen=True)
class InVitroMeasurement:
    """One primary in-vitro measurement, with everything needed to interpret it.

    `not_printed` is as load-bearing as the value. A Ki whose sodium
    concentration was never printed cannot be compared to one whose was, and
    listing the gaps stops a later reader from assuming the blank fields were
    checked and found equal.
    """

    label: str
    value: float
    value_kind: str
    value_as_printed: str
    printed_or_converted: str
    ligand: str
    isoform: str
    preparation: str
    tracer: str
    tracer_class: str
    sodium: str
    guanine_nucleotide: str
    temperature: str
    incubation_time: str
    nonspecific_binding: str
    replicates: str
    citation: str
    pmid: str
    access_route: str
    not_printed: tuple = ()
    notes: str = ""

    @property
    def is_binding_constant(self):
        return self.value_kind in {"Ki", "KD"}


#: Displacement binding constants at cloned human D2, lowest first.
HALOPERIDOL_HUMAN_D2_BINDING = (
    InVitroMeasurement(
        label="roberts2005",
        value=0.13,
        value_kind="Ki",
        value_as_printed="pKi(+GTP) 9.90 +/- 0.04 (0.13 nM), n = 4",
        printed_or_converted="printed in nM alongside the pK",
        ligand="haloperidol",
        isoform="D2S",
        preparation="membranes from CHO cells stably expressing human D2short, 25 ug membrane protein per 1 ml assay",
        tracer="[3H]NPA, 1 nM in competition",
        tracer_class="agonist radioligand",
        sodium="none: 100 mM N-methyl-D-glucamine substituted for sodium, in 20 mM HEPES, 1 mM EGTA, 1 mM EDTA, 10 mM MgCl2, 0.1 mM DTT, pH 7.4",
        guanine_nucleotide="GTP present for the registered value; the paper also reports the no-GTP condition",
        temperature="25 C",
        incubation_time="3 h",
        nonspecific_binding="in competition, the maximal inhibition of [3H]NPA binding by the ligand; in saturation, 3 uM (+)-butaclamol",
        replicates="n = 4",
        citation="Roberts DJ, Strange PG. Br J Pharmacol. 2005;145(1):34-42. doi:10.1038/sj.bjp.0706073",
        pmid="15735658",
        access_route="Europe PMC rendered PDF, PMC1576109",
        not_printed=("Kd of [3H]NPA in this preparation", "Bmax"),
        notes=(
            "This paper prints three constants for haloperidol, not one: "
            "K_uncoupled 0.13 nM, K_coupled 0.17 nM, and Ki with GTP 0.13 nM. The "
            "GTP condition is registered here because it is the G-protein-uncoupled "
            "state an antagonist affinity is normally quoted for. Transcription "
            "note: pdftotext renders this PDF's micro sign as 'm' and its plus-minus "
            "as '7', so every micro-prefixed quantity above is a corrected reading."
        ),
    ),
    InVitroMeasurement(
        label="sykes2017",
        value=0.33,
        value_kind="Ki",
        value_as_printed="pKi 9.48 +/- 0.08 s.e.m., four experiments performed in singlet",
        printed_or_converted="back-converted from the printed pKi; the paper prints logs only",
        ligand="haloperidol",
        isoform="D2L",
        preparation="SNAP-tagged human D2L (NM_000795) in a stable dilution-cloned CHO-D2L line, 2 ug membranes per well, 40 ul assay volume",
        tracer="PPHT-red, 12.5 nM",
        tracer_class="fluorescent agonist, TR-FRET",
        sodium="138 mM NaCl in 20 mM HEPES, 6 mM MgCl2, 1 mM EGTA, 1 mM EDTA, 0.02% pluronic acid, pH 7.4",
        guanine_nucleotide="100 uM GppNHp, to ensure binding only to the G-protein-uncoupled receptor",
        temperature="37 C",
        incubation_time="not printed for the equilibrium competition",
        nonspecific_binding="10 uM haloperidol",
        replicates="four experiments in singlet",
        citation="Sykes DA, Moore H, Stott L, et al. Nat Commun. 2017;8(1):763. doi:10.1038/s41467-017-00716-z",
        pmid="28970469",
        access_route="Europe PMC fullTextXML, PMC5624946",
        not_printed=("incubation time", "Bmax", "ligand-depletion correction"),
        notes=(
            "The tracer is a fluorescent agonist, not a radioligand, so this is a "
            "different measurement class from the filtration assays. The paper does "
            "run a separate [3H]spiperone saturation, but the registered value does "
            "not come from it. IC50 was converted to Ki by Cheng-Prusoff."
        ),
    ),
    InVitroMeasurement(
        label="fan2020",
        value=0.46,
        value_kind="Ki",
        value_as_printed="0.46 nM (pKi 9.34 +/- 0.09 SEM), n = 3 independent experiments",
        printed_or_converted="printed in nM in Supplementary Table 2",
        ligand="haloperidol",
        isoform="D2L",
        preparation="HEK293T (ATCC CRL-11268) membranes transiently expressing DRD2 long isoform, pcDNA3.1",
        tracer="[3H]-N-methylspiperone, 0.8-1.0 nM in competition",
        tracer_class="antagonist radioligand, filtration",
        sodium="none in the printed recipe: 50 mM Tris, 10 mM MgCl2, 0.1 mM EDTA, 0.1% BSA, pH 7.4",
        guanine_nucleotide="not printed",
        temperature="room temperature in the dark, not printed in degrees",
        incubation_time="2 h",
        nonspecific_binding="10 uM haloperidol, printed for the saturation experiments only, not for the competition that produced this value",
        replicates="n = 3",
        citation="Fan L, Tan L, Chen Z, et al. Nat Commun. 2020;11(1):1074, Supplementary Table 2. doi:10.1038/s41467-020-14884-y",
        pmid="32103023",
        access_route="Europe PMC / NCBI efetch for the body, PMC7044277; the value itself from the free Supplementary PDF (MOESM1) at static-content.springer.com",
        not_printed=("temperature in degrees", "Kd or Bmax for the radioligand at wild-type DRD2", "membrane protein per well", "nonspecific binding for the competition assay"),
        notes=(
            "The value is not in the article body, only in Supplementary Table 2. "
            "The membrane-preparation sentence does not itself say 'human'; the word "
            "appears for the identically named construct in the BRET methods and in "
            "the Supplementary Table 1 header, and it is registered as human on that "
            "basis rather than on the binding methods sentence alone."
        ),
    ),
    InVitroMeasurement(
        label="hall1997",
        value=2.9,
        value_kind="Ki",
        value_as_printed="pKi 8.54 +/- 0.04 s.e.mean (2.9 nM), n = 3",
        printed_or_converted="printed in a separate Ki (nM) column",
        ligand="haloperidol",
        isoform="D2S",
        preparation="membranes from CHO cells expressing human D2short, 10 ug membrane protein per assay, receptor density 3.5 +/- 0.4 pmol/mg",
        tracer="[phenyl-4-3H]spiperone, 15-30 Ci/mmol; its Kd in this preparation is printed as 126 pM",
        tracer_class="antagonist radioligand, filtration",
        sodium="100 mM NaCl added, explicitly because sodium is known to affect the affinity of some antipsychotics; base buffer composition deferred to Castro & Strange 1993 and not printed",
        guanine_nucleotide="100 uM GTP",
        temperature="not printed",
        incubation_time="1 h, extended from 45 min to ensure complete equilibration",
        nonspecific_binding="not printed, deferred to Castro & Strange 1993",
        replicates="n = 3",
        citation="Hall DA, Strange PG. Br J Pharmacol. 1997;121(4):731-736. doi:10.1038/sj.bjp.0701196",
        pmid="9208141",
        access_route="Europe PMC rendered PDF, PMC1564749",
        not_printed=("radioligand concentration in the competition assay", "base buffer composition", "temperature", "nonspecific-binding definition"),
        notes=(
            "Run under both high sodium and G-protein-uncoupling conditions. "
            "Transcription note: as with roberts2005, pdftotext renders this PDF's "
            "micro sign as 'm', so the 100 uM GTP and 10 ug figures are corrected "
            "readings."
        ),
    ),
    InVitroMeasurement(
        label="kaczor2023a",
        value=6.56,
        value_kind="Ki",
        value_as_printed="6.56 +/- 0.88 nM, mean +/- SEM of 2-3 independent experiments in duplicate",
        printed_or_converted="printed in nM",
        ligand="haloperidol",
        isoform="D2S",
        preparation="CHO-K1 cells with stable expression of the cloned human D2S receptor, 250 ul assay volume",
        tracer="[3H]spiperone, 0.6 nM",
        tracer_class="antagonist radioligand, filtration",
        sodium="not printed in the article body",
        guanine_nucleotide="not printed in the article body",
        temperature="not printed in the article body",
        incubation_time="not printed in the article body",
        nonspecific_binding="100 uM sulpiride",
        replicates="2-3 independent experiments in duplicate",
        citation="Kaczor AA, et al. Molecules. 2023;28(11):4211, Table 1. doi:10.3390/molecules28114211",
        pmid="37241951",
        access_route="Europe PMC fullTextXML, PMC10223470; the Supplementary Table S2 holding the conditions returned HTTP 403 from the publisher and is unretrieved",
        not_printed=("buffer composition", "sodium", "guanine nucleotide", "temperature", "incubation time", "Kd of [3H]spiperone"),
        notes=(
            "Transcription trap: this paper's Table 2 haloperidol row, 6.31 and 4.95 "
            "nM, is human D1 and D3, not D2."
        ),
    ),
    InVitroMeasurement(
        label="kaczor2023b",
        value=7.41,
        value_kind="Ki",
        value_as_printed="pKi 8.13 +/- 0.05 (7.41 nM), n = 11",
        printed_or_converted="printed in nM alongside the pKi",
        ligand="haloperidol",
        isoform="D2S",
        preparation="CHO-K1 cells with stable expression of the cloned human D2S receptor",
        tracer="[3H]spiperone, 0.6 nM",
        tracer_class="antagonist radioligand, filtration",
        sodium="not printed in the article body",
        guanine_nucleotide="not printed in the article body",
        temperature="not printed in the article body",
        incubation_time="not printed in the article body",
        nonspecific_binding="100 uM sulpiride",
        replicates="n = 11",
        citation="Kaczor AA, et al. J Enzyme Inhib Med Chem. 2023;38(1):2200358, Table 1. doi:10.1080/14756366.2023.2200358",
        pmid="37184096",
        access_route="Europe PMC fullTextXML, PMC10187112; Supplementary Table S2 unretrieved",
        not_printed=("buffer composition", "sodium", "guanine nucleotide", "temperature", "incubation time", "Kd of [3H]spiperone"),
        notes=(
            "An aggregated reference-compound value across 11 runs. Same Table 2 trap "
            "as kaczor2023a: the 4.95 and 5.81 nM there are D1 and D3."
        ),
    ),
    InVitroMeasurement(
        label="bartuzi2022",
        value=8.85,
        value_kind="Ki",
        value_as_printed="Ki value for haloperidol of 8.85 nM",
        printed_or_converted="printed in nM",
        ligand="haloperidol",
        isoform="D2S",
        preparation="membranes from CHO-K1 cells stably expressing the cloned human D2S receptor",
        tracer="[3H]spiperone, 1.5 nM (76.1 Ci/mmol)",
        tracer_class="antagonist radioligand, filtration",
        sodium="120 mM NaCl in 50 mM Tris-HCl, 5 mM KCl, 5 mM MgCl2, 1 mM EDTA, pH 7.4",
        guanine_nucleotide="not printed",
        temperature="25 C",
        incubation_time="120 min",
        nonspecific_binding="10 uM sulpiride",
        replicates="two independent experiments in duplicate, stated for the figure rather than for the value",
        citation="Bartuzi D, Kaczor AA, Matosiuk D, et al. Pharmacol Rep. 2022;74(2):406-424. doi:10.1007/s43440-021-00352-x",
        pmid="35064921",
        access_route="Europe PMC fullTextXML, PMC8964653",
        not_printed=("Kd of [3H]spiperone in this preparation", "error or n on the Ki itself", "Bmax", "guanine nucleotide"),
        notes=(
            "The only source found that prints radioligand and its concentration, "
            "human isoform, sodium, temperature, incubation time and the "
            "nonspecific-binding definition together. Note that the paper's title "
            "concerns D2L while its binding assay uses D2S."
        ),
    ),
)

#: Functional antagonist potencies at human D2, kept apart from the binding
#: constants because they are a different measurement. Both come from papers that
#: also report a binding Ki, which is what makes the within-paper divergence
#: visible at all.
HALOPERIDOL_HUMAN_D2_FUNCTIONAL = (
    InVitroMeasurement(
        label="kaczor2023a-cAMP",
        value=0.11,
        value_kind="Kb",
        value_as_printed="pKb 9.97 +/- 0.22 (0.11 nM)",
        printed_or_converted="printed in nM alongside the pKb",
        ligand="haloperidol",
        isoform="not stated in the table caption",
        preparation="cells stably overexpressing the cloned human receptor",
        tracer="not applicable",
        tracer_class="functional, inhibition of forskolin-stimulated cAMP production",
        sodium="not printed",
        guanine_nucleotide="not applicable",
        temperature="not printed",
        incubation_time="not printed",
        nonspecific_binding="not applicable",
        replicates="not printed",
        citation="Kaczor AA, et al. Molecules. 2023;28(11):4211, Table 3. doi:10.3390/molecules28114211",
        pmid="37241951",
        access_route="Europe PMC fullTextXML, PMC10223470",
        not_printed=("isoform", "temperature", "incubation time", "constitutive activity of the system"),
        notes="Sixty-fold below the same paper's own binding Ki of 6.56 nM.",
    ),
    InVitroMeasurement(
        label="kaczor2023b-cAMP",
        value=0.29,
        value_kind="Kb",
        value_as_printed="pKb 9.53 +/- 0.07 (0.29 nM), 92.1 +/- 0.6% inhibition of the dopamine response",
        printed_or_converted="printed in nM alongside the pKb",
        ligand="haloperidol",
        isoform="not stated in the table caption",
        preparation="cells stably overexpressing the cloned human receptor",
        tracer="not applicable",
        tracer_class="functional, inhibition of the dopamine response",
        sodium="not printed",
        guanine_nucleotide="not applicable",
        temperature="not printed",
        incubation_time="not printed",
        nonspecific_binding="not applicable",
        replicates="not printed",
        citation="Kaczor AA, et al. J Enzyme Inhib Med Chem. 2023;38(1):2200358, Table 3. doi:10.1080/14756366.2023.2200358",
        pmid="37184096",
        access_route="Europe PMC fullTextXML, PMC10187112",
        not_printed=("isoform", "temperature", "incubation time", "constitutive activity of the system"),
        notes=(
            "Twenty-six-fold below the same paper's own binding Ki of 7.41 nM. Its "
            "92.1% figure is inhibition of an agonist response, which does not by "
            "itself separate a neutral antagonist from an inverse agonist."
        ),
    ),
)


@dataclass(frozen=True)
class IntrinsicEfficacy:
    """A measured intrinsic efficacy, on the scale documented in EFFICACY_SCALE."""

    label: str
    efficacy: float
    classification: str
    value_as_printed: str
    readout: str
    reference_compound: str
    preparation: str
    constitutive_activity: str
    conditions: str
    replicates: str
    citation: str
    pmid: str
    access_route: str
    caveats: tuple = ()


#: Haloperidol is an inverse agonist at human D2, not a neutral antagonist.
#: Two independent functional readouts agree on the direction and on a high
#: relative efficacy. Both are from the same laboratory, which is a real
#: independence limitation and is recorded in the caveats rather than left for a
#: reader to notice.
HALOPERIDOL_HUMAN_D2_INTRINSIC_EFFICACY = (
    IntrinsicEfficacy(
        label="hall1997-cAMP",
        efficacy=-0.95,
        classification="inverse agonist, full by this assay's own reference",
        value_as_printed="maximal effect 0.95 +/- 0.03 relative to 1 uM (+)-butaclamol; pEC50 8.69 +/- 0.08 (2.1 nM); n = 3",
        readout="forskolin-stimulated cyclic AMP accumulation in whole cells",
        reference_compound="(+)-butaclamol at 1 uM, a full curve run with every concentration-response curve",
        preparation="CHO cells expressing human D2short, receptor density 3.5 +/- 0.4 pmol/mg",
        constitutive_activity=(
            "(+)-butaclamol raised forskolin-stimulated cAMP by about 150% above "
            "control in CHO-D2S and about 180% in CHO-D2L"
        ),
        conditions="whole cells; the paper reports no significant difference between haloperidol's maximal effect and the butaclamol control",
        replicates="n = 3",
        citation="Hall DA, Strange PG. Br J Pharmacol. 1997;121(4):731-736, Table 1. doi:10.1038/sj.bjp.0701196",
        pmid="9208141",
        access_route="Europe PMC rendered PDF, PMC1564749",
        caveats=(
            "High heterologous receptor expression, which is what makes the "
            "constitutive activity measurable in the first place.",
            "Referenced to (+)-butaclamol, not to the endogenous agonist; see "
            "EFFICACY_SCALE.",
            "Same laboratory as roberts2005-GTPgS, so the two are not independent "
            "of each other in the way two labs would be.",
        ),
    ),
    IntrinsicEfficacy(
        label="roberts2005-GTPgS",
        efficacy=-0.83,
        classification="inverse agonist, high relative efficacy",
        value_as_printed="83 +/- 6 % of the maximal (+)-butaclamol response; pIC50 9.54 +/- 0.10 (0.29 nM); n = 5",
        readout="[35S]GTPgammaS binding in membranes",
        reference_compound="(+)-butaclamol, full curve in every experiment",
        preparation="membranes from CHO cells stably expressing human D2short, 25 ug protein in 0.9 ml",
        constitutive_activity=(
            "the tested compounds inhibited [35S]GTPgammaS binding by 15-20%; the "
            "paper does not express this as a fold-over-basal or a percent of "
            "receptor active"
        ),
        conditions="HEPES buffer with 100 mM N-methyl-D-glucamine and no added GDP unless specified, 20 min at 30 C, then 100 pM [35S]GTPgammaS for 30 min",
        replicates="n = 5",
        citation="Roberts DJ, Strange PG. Br J Pharmacol. 2005;145(1):34-42, Table 2. doi:10.1038/sj.bjp.0706073",
        pmid="15735658",
        access_route="Europe PMC rendered PDF, PMC1576109",
        caveats=(
            "The constitutive activity was deliberately maximised: the authors omit "
            "sodium to increase receptor/G-protein coupling and note that inverse "
            "agonist signals were greater with no or low GDP. The 83% figure is "
            "therefore condition-specific, not a constant.",
            "Referenced to (+)-butaclamol; see EFFICACY_SCALE.",
            "Same laboratory as hall1997-cAMP.",
        ),
    ),
)

#: Independent qualitative confirmation of the direction, from a lab unconnected
#: to the two quantitative sources. Abstract only, so it carries no number and is
#: not in HALOPERIDOL_HUMAN_D2_INTRINSIC_EFFICACY.
INVERSE_AGONISM_ABSTRACT_ONLY_SUPPORT = (
    "Burstein ES, Ma J, Wong S, et al. J Pharmacol Exp Ther. 2005;315(3):1278-1287. "
    "PMID 16135699. Paywalled; publisher returns 403 from this host and there is no "
    "PMCID. The abstract reports that overexpression of Galpha-o induces constitutive "
    "activity in the human D2-like receptors and that virtually all the antipsychotics "
    "tested were D2 and D3 inverse agonists, haloperidol among them. The functional "
    "readout is not named in the abstract and no efficacy value is printed, so this is "
    "usable only as a sourced qualitative statement of direction."
)

#: Sources examined and excluded, each for a stated reason.
EXCLUDED_SOURCES = (
    (
        "de Witte WEA, et al. Br J Pharmacol. 2018;175(21):4121-4136. PMID 30051456",
        "KD 0.382 nM (SD 0.050) by competition kinetics with a PPHT-based fluorescent "
        "agonist tracer. Excluded because the paper does not state that the receptor "
        "in the binding assay is human; only its separate cAMP assay is explicitly "
        "human. It is also a kinetic KD rather than a Ki.",
    ),
    (
        "McCormick PN, et al. Neuropsychopharmacology. 2010;35(8):1826-1835. PMID 20410873",
        "Rat brain, in the paper's own words 'in vitro and ex vivo [3H]-(+)-PHNO "
        "binding sites in rat brain'. Not usable under a human label.",
    ),
    (
        "Kroeze WK, et al. Neuropsychopharmacology. 2003. PMID 12629531",
        "No PMCID and Europe PMC reports it closed. Unpaywall lists bronze open "
        "access at a publisher URL that returns bot-blocked HTML from this host, and "
        "its only repository location redirects to a BindingDB entry page rather than "
        "the article. Paywalled, unavailable, no number taken.",
    ),
    (
        "Seeman P, Van Tol HHM. 1995",
        "Paywalled. Cited for the effect it reported, with no number from it used "
        "anywhere in this module.",
    ),
    (
        "PDSP Ki database",
        "An aggregator over primary sources rather than a primary source. Excluded so "
        "that every value here carries its own printed assay conditions.",
    ),
)

#: Searched and empty, recorded as a result rather than left as a silent gap.
SEARCHED_AND_EMPTY = (
    (
        "human tissue D2 affinity",
        "No primary human post-mortem brain, caudate, striatum or pituitary "
        "haloperidol D2 Ki was retrievable. Those searches returned reviews and "
        "conference-abstract collections only. Every value above is from a "
        "heterologous cell line.",
    ),
    (
        "beta-arrestin intrinsic efficacy at human D2",
        "No primary source measuring haloperidol's intrinsic efficacy at human D2 by "
        "beta-arrestin recruitment was retrievable. Thirty open-access records were "
        "screened; none was such a measurement. Constitutive beta-arrestin "
        "recruitment at D2 is not routinely measurable, so this may be a real absence "
        "in the literature rather than a retrieval failure. Only the non-retrieval is "
        "claimed here.",
    ),
)

#: Which binding record the receptor registry cites when a Ki is called for.
#: Chosen on completeness of printed conditions, not on the value: a Ki whose
#: sodium, temperature and incubation time were never printed cannot be
#: interpreted or compared against another. It happens to be the highest of the
#: seven, so the choice does not flatter the in-vitro to in-vivo agreement.
REGISTRY_KI_LABEL = "bartuzi2022"

#: Which efficacy record the receptor registry cites. The cAMP measurement is
#: taken over the [35S]GTPgammaS one because it is a whole-cell readout, one step
#: closer to the downstream effect the model represents, and because its authors
#: report haloperidol's maximum as statistically indistinguishable from their
#: reference compound's. The other value, -0.83, is registered alongside it and
#: the difference between them is reported rather than averaged away.
REGISTRY_EFFICACY_LABEL = "hall1997-cAMP"


def haloperidol_d2_binding(label=None):
    if label is None:
        return HALOPERIDOL_HUMAN_D2_BINDING
    return tuple(r for r in HALOPERIDOL_HUMAN_D2_BINDING if r.label == label)


def registry_ki():
    """The single binding record the D2 receptor parameter cites."""
    records = haloperidol_d2_binding(REGISTRY_KI_LABEL)
    if not records:
        raise ValueError("REGISTRY_KI_LABEL %r is not a registered record" % REGISTRY_KI_LABEL)
    return records[0]


def registry_efficacy():
    """The single efficacy record the D2 receptor parameter cites."""
    for record in HALOPERIDOL_HUMAN_D2_INTRINSIC_EFFICACY:
        if record.label == REGISTRY_EFFICACY_LABEL:
            return record
    raise ValueError("REGISTRY_EFFICACY_LABEL %r is not a registered record" % REGISTRY_EFFICACY_LABEL)


def binding_ki_range_nM():
    values = [r.value for r in HALOPERIDOL_HUMAN_D2_BINDING]
    return min(values), max(values)


def binding_ki_fold_spread():
    low, high = binding_ki_range_nM()
    return high / low


def efficacy_range():
    values = [r.efficacy for r in HALOPERIDOL_HUMAN_D2_INTRINSIC_EFFICACY]
    return min(values), max(values)


def in_vitro_in_vivo_ratios(ec50_nM=None):
    """Ratio of the in-vivo EC50 to each in-vitro binding Ki.

    Returned as the full set rather than as a single number. A single ratio would
    have to pick one Ki out of the seven, and that choice moves the answer by the
    same 68-fold the Ki set spans.
    """
    if ec50_nM is None:
        from .d2_ec50 import HALOPERIDOL_D2_PLASMA_EC50_nM

        ec50_nM = HALOPERIDOL_D2_PLASMA_EC50_nM
    return tuple(
        (record.label, ec50_nM / record.value)
        for record in HALOPERIDOL_HUMAN_D2_BINDING
    )


def in_vitro_in_vivo_ratio_range(ec50_nM=None):
    ratios = [value for _, value in in_vitro_in_vivo_ratios(ec50_nM)]
    return min(ratios), max(ratios)


def ratio_straddles_unity(ec50_nM=None):
    """True when the in-vitro Ki set brackets the in-vivo EC50.

    When it does, the in-vitro measurements cannot even agree on whether the
    in-vivo constant is larger or smaller than they are, so no single correction
    factor could be applied to a Ki to obtain an EC50.
    """
    low, high = in_vitro_in_vivo_ratio_range(ec50_nM)
    return low < 1.0 < high
