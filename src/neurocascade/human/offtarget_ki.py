"""P23.1: human off-target binding fingerprints for the Phase 23 thesis test.

Phase 23 asks the one question this whole project was built to answer: does a
drug's off-target *binding fingerprint* explain a clinical difference that equal
D2 occupancy cannot? The dataset that forces the question is Siafis 2023
(PMC10618092, N=37,193): haloperidol at 11 mg/day carries an extrapyramidal odds
ratio of 4.7, risperidone at 16 mg/day an OR of 3.3, at *approximately the same*
~87% striatal D2 occupancy. Occupancy alone does not determine the outcome. If
anything in the model can, it is the rest of the receptor profile.

This module is the input layer for that test: per-drug **human** equilibrium
dissociation constants (Ki) at the five off-targets Phase 23 names -- 5-HT2A, M1,
H1, alpha-1 adrenoceptor and dopamine D3 -- for the two drugs the Siafis
contrast is built on, haloperidol and risperidone. It makes no prediction; it
assembles the numbers a later Phase 23 task will compare. By rule 1 of the
validation ledger that earns no validated-claims row, and this module claims
none.

Source and provenance
---------------------
Every value is transcribed from the **PDSP Ki Database** (the Psychoactive Drug
Screening Program, Roth lab, UNC), downloaded in full as CSV from
https://pdsp.unc.edu/databases/kiDownload/ on 2026-10-01 (98,678 rows). The task
brief names PDSP as the source; it is free, public, and -- unlike a review table
-- carries the **species** and the **primary reference** on every row, which is
exactly what the species gate needs. Each registered record keeps the PDSP
`species`, `source` (assay preparation) and `Reference` fields verbatim, plus a
resolved PMID for the three load-bearing primary papers.

`neurocascade.human.d2_in_vitro` deliberately *excluded* PDSP for the D2 Ki,
because for D2 seven primaries could be retrieved in full text with their assay
conditions and PDSP would only have flattened them. For these five off-targets
no such full-text retrieval was done; PDSP is the primary access route here, and
that difference is stated rather than hidden. The limitation is real: a PDSP row
does not carry the sodium, tracer and temperature that `d2_in_vitro` records,
and the within-cell spread below (up to ~17-fold across human measurements of
one drug at one target) is the visible cost of that.

The species gate, used not re-implemented
-----------------------------------------
PDSP mixes species freely. Schotte 1996 (PMID 8935801) -- the canonical
risperidone binding paper -- is the clearest example: it measured 5-HT2A, D2 and
D3 on cloned **human** receptors but alpha-1 on **rat** cortex and H1 on
**guinea-pig** cerebellum, all in one table. A naive read would carry the rodent
numbers straight into a human fingerprint. `gate_record` routes each record's
provenance species through the *existing* `species_gate.gate`, the same function
every human prediction already passes through: human records return (pass),
every non-human one raises `UnsourcedParameterError`. `NON_HUMAN_EXCLUDED` holds
a representative set of those refused rows (rat, guinea-pig, pig, bovine, mouse)
so the exclusion is demonstrated on real data, not asserted.

Selection rule (stated so it cannot be read as cherry-picking)
-------------------------------------------------------------
Each (drug, target) cell has many human PDSP rows (8 to 30), most of them
secondary re-tabulations of the same primaries, several stored as pKi
back-conversions (the long-decimal values below). Registering all of them would
dress duplication as independence. Instead, for each cell the registry holds a
fixed, outcome-independent set:

  * the human **tissue** value (Richelson & Souder 2000, PMID 11132243, human
    post-mortem brain) where one exists -- the gold-standard provenance for this
    project, available for 5-HT2A, H1 and alpha-1 but **not** for D3 or M1;
  * the **minimum and maximum** human value in the PDSP download for that cell,
    each with its own reference, so `human_ki_range_nM` reports the true full
    span rather than a flattering subset;
  * the canonical antipsychotic paper, **Schotte 1996**, where it measured the
    target on a human receptor;
  * **Kroeze 2003** (PMID 12629531, Roth lab, the lab that runs PDSP), where
    present;
  * for D3 and M1, one classic human primary each (Sokoloff 1992 for D3;
    Bolden 1992 for risperidone M1).

The headline
-----------
All ten cells (two drugs x five targets) are sourced to a human value; none is
searched-and-empty at the target level. Two sub-class absences *are* recorded in
`SEARCHED_AND_EMPTY`: there is no human-**tissue** D3 or M1 Ki for either drug in
PDSP, only cloned/recombinant. And M1 is a near-miss for both drugs -- the human
values are dominated by the >10000 nM assay ceiling, with the few finite numbers
in the 1.6-2.8 uM range: neither haloperidol nor risperidone has clinically
meaningful muscarinic affinity, so M1 cannot be part of whatever distinguishes
their EPS. That is itself a result the Phase 23 test needs, recorded here rather
than discovered later.
"""
from dataclasses import dataclass

from .. import species_gate

#: When the PDSP full CSV was downloaded.
PDSP_RETRIEVED = "2026-10-01"

#: How every value here was reached. Each record's `reference` is the primary as
#: PDSP curates it; this is the access route to it.
PDSP_SOURCE = (
    "PDSP Ki Database full CSV, https://pdsp.unc.edu/databases/kiDownload/ "
    "(downloaded 2026-10-01, 98,678 rows)"
)

#: The five off-targets Phase 23 names, canonical tokens matching fingerprint.TARGETS.
TARGETS = ("5HT2A", "M1", "H1", "alpha1", "D3")

#: The two drugs the Siafis 2023 contrast is built on.
DRUGS = ("haloperidol", "risperidone")

#: PMIDs resolved for the load-bearing primaries. A PDSP reference string with no
#: entry here carries author+year only, as PDSP prints it; no PMID is invented.
_PMIDS = {
    "Richelson E, Souder T. 2000": "11132243",
    "Schotte A, et al.,  1996": "8935801",
    "Kroeze WK, et.al., 2003": "12629531",
}

#: The Siafis 2023 numbers this phase exists to explain. Carried as text, not as
#: a computed input: nothing here consumes them, they state why the fingerprint
#: is being assembled at all.
SIAFIS_2023 = (
    "Siafis et al. 2023, PMC10618092, N=37,193: haloperidol ~11 mg/day EPS "
    "odds ratio 4.7 vs risperidone ~16 mg/day OR 3.3, both at ~87% striatal D2 "
    "occupancy. Equal occupancy, unequal extrapyramidal risk."
)


@dataclass(frozen=True)
class OffTargetKi:
    """One PDSP Ki row for one drug at one off-target, provenance intact.

    `value_nM` and `value_as_printed` are both kept because several PDSP rows are
    pKi back-conversions (the long-decimal values); the printed form says so, so
    a later reader does not mistake a converted number for a measured precision.
    `censored` carries the PDSP ki-note for assay-ceiling rows (">"), which are a
    bound, not a point measurement -- the M1 ceilings especially.
    """

    drug: str
    target: str
    value_nM: float
    censored: str          # '', '>' or '<', the PDSP ki-note
    species: str           # PDSP species token, verbatim ('HUMAN','RAT','GUINEA PIG',...)
    preparation: str       # PDSP source/assay field, verbatim ('BRAIN','CLONED','FR. CORTEX',...)
    reference: str         # PDSP reference field, verbatim
    value_as_printed: str
    selection: str         # why this row is registered (tissue / min / max / canonical / ...)
    pmid: str = ""
    note: str = ""

    @property
    def is_human(self):
        return species_gate.normalize_species(self.species) == species_gate.HUMAN

    @property
    def provenance_class(self):
        if not self.is_human:
            return "non-human (excluded by species gate)"
        prep = self.preparation.upper()
        if any(t in prep for t in ("BRAIN", "CORTEX", "CAUDATE")):
            return "human tissue"
        if "CLONED" in prep or "RECOMBINANT" in prep:
            return "human recombinant"
        return "human (preparation unspecified)"


def _r(drug, target, value, species, prep, ref, selection, *, censored="",
       printed=None, note=""):
    return OffTargetKi(
        drug=drug, target=target, value_nM=float(value), censored=censored,
        species=species, preparation=prep, reference=ref,
        value_as_printed=printed or ("%s%g nM" % (censored, value)),
        selection=selection, pmid=_PMIDS.get(ref, ""), note=note,
    )


# ---------------------------------------------------------------------------
# Human records, by the selection rule in the module docstring.
# ---------------------------------------------------------------------------

HUMAN_KI = (
    # --- haloperidol ------------------------------------------------------
    # 5-HT2A: human tissue 61; human cloned 25-435.
    _r("haloperidol", "5HT2A", 61, "HUMAN", "BRAIN",
       "Richelson E, Souder T. 2000", "human tissue (post-mortem brain)"),
    _r("haloperidol", "5HT2A", 25, "HUMAN", "CLONED",
       "SEEMAN P, ET AL., 1997", "human-cloned minimum"),
    _r("haloperidol", "5HT2A", 435, "HUMAN", "CLONED",
       "Fernandez J, et al., 2005", "human-cloned maximum"),
    _r("haloperidol", "5HT2A", 200, "HUMAN", "CLONED",
       "Schotte A, et al.,  1996", "canonical antipsychotic paper (human cloned)"),
    _r("haloperidol", "5HT2A", 53, "HUMAN", "CLONED",
       "Kroeze WK, et.al., 2003", "Roth-lab / PDSP paper"),
    # M1: no human tissue; cloned ceiling-dominated, finite tail 1585-1600.
    _r("haloperidol", "M1", 10000, "HUMAN", "CLONED", "PDSP Certified Data",
       "human-cloned ceiling (assay bound)", censored=">",
       printed=">10000 nM (assay ceiling)"),
    _r("haloperidol", "M1", 1584.8931924611, "HUMAN", "CLONED",
       "Schmidt, AW. et, al. 2001", "human-cloned minimum (finite)",
       printed="1585 nM (PDSP pKi back-conversion)"),
    _r("haloperidol", "M1", 1600, "HUMAN", "CLONED",
       "Hirose, Tsuyoshi.  et. al, 2005", "human-cloned, finite confirmatory"),
    # H1: human tissue 260; cloned 215-4160.
    _r("haloperidol", "H1", 260, "HUMAN", "BRAIN",
       "Richelson E, Souder T. 2000", "human tissue (post-mortem brain)"),
    _r("haloperidol", "H1", 215, "HUMAN", "CLONED",
       "Ghoneim OM, et al., 2006", "human-cloned minimum"),
    _r("haloperidol", "H1", 4160, "HUMAN", "CLONED",
       "Maheux J, et al., 2005", "human-cloned maximum"),
    _r("haloperidol", "H1", 790, "HUMAN", "CLONED",
       "Schotte A, et al.,  1996", "canonical antipsychotic paper (human cloned)"),
    _r("haloperidol", "H1", 1800, "HUMAN", "CLONED",
       "Kroeze WK, et.al., 2003", "Roth-lab / PDSP paper"),
    # alpha-1: human tissue 17; cloned 4.68-17.9.
    _r("haloperidol", "alpha1", 17, "HUMAN", "BRAIN",
       "Richelson E, Souder T. 2000", "human tissue (post-mortem brain)"),
    _r("haloperidol", "alpha1", 4.677351412872, "HUMAN", "CLONED",
       "Schmidt, AW. et, al. 2001", "human-cloned minimum",
       printed="4.68 nM (PDSP pKi back-conversion)"),
    _r("haloperidol", "alpha1", 17.9, "HUMAN", "CLONED",
       "Ishibashi, Tadashi.  et. al, 2010", "human-cloned maximum"),
    _r("haloperidol", "alpha1", 12, "HUMAN", "CLONED",
       "Kroeze WK, et.al., 2003", "Roth-lab / PDSP paper (alpha1A)",
       note="PDSP target adrenergic Alpha1A"),
    # D3: no human tissue; cloned 0.2-23.
    _r("haloperidol", "D3", 0.2, "HUMAN", "CLONED",
       "Burstein ES, et al., 2005", "human-cloned minimum"),
    _r("haloperidol", "D3", 23, "HUMAN", "CLONED",
       "Seeman, Philip  2001", "human-cloned maximum"),
    _r("haloperidol", "D3", 7.8, "HUMAN", "CLONED",
       "Schotte A, et al.,  1996", "canonical antipsychotic paper (human cloned)"),
    _r("haloperidol", "D3", 2.9, "HUMAN", "CLONED",
       "Sokoloff P, et al., 1992", "classic human D3 primary"),

    # --- risperidone ------------------------------------------------------
    # 5-HT2A: human tissue 0.15; cloned 0.14-7.
    _r("risperidone", "5HT2A", 0.15, "HUMAN", "BRAIN",
       "Richelson E, Souder T. 2000", "human tissue (post-mortem brain)"),
    _r("risperidone", "5HT2A", 0.14, "HUMAN", "CLONED",
       "SEEMAN P, ET AL., 1997", "human-cloned minimum"),
    _r("risperidone", "5HT2A", 7, "HUMAN", "CLONED",
       "Bymaster F.P., et. al., 1999", "human-cloned maximum"),
    _r("risperidone", "5HT2A", 0.52, "HUMAN", "CLONED",
       "Schotte A, et al.,  1996", "canonical antipsychotic paper (human cloned)"),
    _r("risperidone", "5HT2A", 0.17, "HUMAN", "CLONED",
       "Kroeze WK, et.al., 2003", "Roth-lab / PDSP paper"),
    # M1: no human tissue; ceiling-dominated, finite tail 2800-2818.
    _r("risperidone", "M1", 10000, "HUMAN", "CLONED", "Bolden C, et al., 1992",
       "classic human M1 primary (assay bound)", censored=">",
       printed=">10000 nM (assay ceiling)"),
    _r("risperidone", "M1", 2800, "HUMAN", "UNDEFINED",
       "Ablordeppey SY et al., 2008", "human minimum (finite)"),
    _r("risperidone", "M1", 2818.3829312645, "HUMAN", "CLONED",
       "Schmidt, AW. et, al. 2001", "human-cloned, finite confirmatory",
       printed="2818 nM (PDSP pKi back-conversion)"),
    # H1: human tissue 5.2; cloned 3.46-53.7.
    _r("risperidone", "H1", 5.2, "HUMAN", "BRAIN",
       "Richelson E, Souder T. 2000", "human tissue (post-mortem brain)"),
    _r("risperidone", "H1", 3.46, "HUMAN", "CLONED",
       "Ishibashi, Tadashi.  et. al, 2010", "human-cloned minimum"),
    _r("risperidone", "H1", 53.703179637025, "HUMAN", "CLONED",
       "Appl, Heidrun.  et. al, 2011", "human-cloned maximum",
       printed="53.7 nM (PDSP pKi back-conversion)"),
    _r("risperidone", "H1", 27, "HUMAN", "CLONED",
       "Schotte A, et al.,  1996", "canonical antipsychotic paper (human cloned)"),
    _r("risperidone", "H1", 15, "HUMAN", "CLONED",
       "Kroeze WK, et.al., 2003", "Roth-lab / PDSP paper"),
    # alpha-1: human tissue 2.7; cloned 1.38-13.8.
    _r("risperidone", "alpha1", 2.7, "HUMAN", "BRAIN",
       "Richelson E, Souder T. 2000", "human tissue (post-mortem brain)"),
    _r("risperidone", "alpha1", 1.3803842646029, "HUMAN", "CLONED",
       "Schmidt, AW. et, al. 2001", "human-cloned minimum",
       printed="1.38 nM (PDSP pKi back-conversion)"),
    _r("risperidone", "alpha1", 13.803842646029, "HUMAN", "CLONED",
       "Silvestre, JS. et, al. 2005", "human-cloned maximum",
       printed="13.8 nM (PDSP pKi back-conversion)"),
    _r("risperidone", "alpha1", 5, "HUMAN", "CLONED",
       "Kroeze WK, et.al., 2003", "Roth-lab / PDSP paper (alpha1A)",
       note="PDSP target adrenergic Alpha1A"),
    # D3: no human tissue; cloned 1.6-16.
    _r("risperidone", "D3", 1.6, "HUMAN", "CLONED",
       "Burstein ES, et al., 2005", "human-cloned minimum"),
    _r("risperidone", "D3", 16, "HUMAN", "CLONED",
       "Fernandez J, et al., 2005", "human-cloned maximum"),
    _r("risperidone", "D3", 14, "HUMAN", "CLONED",
       "Schotte A, et al.,  1996", "canonical antipsychotic paper (human cloned)"),
)


# ---------------------------------------------------------------------------
# Non-human rows, kept to DEMONSTRATE the gate refusing them on real data.
# Schotte 1996's own table is the centrepiece: human for 5-HT2A/D3, rodent for
# alpha-1, guinea-pig for H1. Pig/bovine/mouse rows from other PDSP sources
# round out the species tokens the gate must refuse.
# ---------------------------------------------------------------------------

NON_HUMAN_EXCLUDED = (
    _r("haloperidol", "5HT2A", 25, "RAT", "FR. CORTEX",
       "Schotte A, et al.,  1996", "non-human (Schotte rat cortex)"),
    _r("haloperidol", "H1", 730, "GUINEA PIG", "CEREBELLUM",
       "Schotte A, et al.,  1996", "non-human (Schotte guinea-pig cerebellum)"),
    _r("haloperidol", "alpha1", 19, "RAT", "Total Cortex",
       "Schotte A, et al.,  1996", "non-human (Schotte rat cortex)"),
    _r("haloperidol", "D3", 21, "RAT", "CLONED",
       "Schotte A, et al.,  1996", "non-human (Schotte rat cloned)"),
    _r("haloperidol", "M1", 1475, "RAT", "CORTEX",
       "Bymaster F.P., et. al., 2003", "non-human (rat cortex)"),
    _r("haloperidol", "D3", 6.5, "BOVINE", "CAUDATE",
       "Schoemaker H et al., 1997", "non-human (bovine caudate)"),
    _r("haloperidol", "D3", 4.3, "MOUSE", "CLONED",
       "Fishburn CS, et al., 1993", "non-human (mouse cloned)"),
    _r("risperidone", "5HT2A", 0.16, "RAT", "FR. CORTEX",
       "Schotte A, et al.,  1996", "non-human (Schotte rat cortex)"),
    _r("risperidone", "H1", 2.6, "GUINEA PIG", "CEREBELLUM",
       "Schotte A, et al.,  1996", "non-human (Schotte guinea-pig cerebellum)"),
    _r("risperidone", "alpha1", 2.3, "RAT", "Total Cortex",
       "Schotte A, et al.,  1996", "non-human (Schotte rat cortex)"),
    _r("risperidone", "D3", 13, "RAT", "CLONED",
       "Schotte A, et al.,  1996", "non-human (Schotte rat cloned)"),
    _r("risperidone", "alpha1", 15.8489319246111, "PIG", "ADRENAL",
       "Wikberg-Matsson A, et al., 1998", "non-human (pig adrenal alpha1A)",
       printed="15.8 nM (PDSP pKi back-conversion)"),
)


#: Searched-and-empty, recorded as a real result (ledger rule 4). The targets
#: themselves are all sourced to human values; what is empty is the human-TISSUE
#: provenance for two of them.
SEARCHED_AND_EMPTY = (
    (
        "human-tissue D3 Ki for haloperidol and risperidone",
        "None in the PDSP download. Every human D3 value for either drug is a "
        "CLONED/recombinant receptor; the only tissue D3 rows are bovine caudate "
        "(Schoemaker 1997, Claustre 2003), which the species gate refuses. "
        "Recorded so the D3 fingerprint is not read as a tissue measurement.",
    ),
    (
        "human-tissue M1 Ki for haloperidol and risperidone",
        "None in the PDSP download. Human M1 values are CLONED or UNDEFINED, and "
        "are dominated by the >10000 nM assay ceiling (the finite ones sit at "
        "1.6-2.8 uM). Neither drug has clinically meaningful muscarinic affinity, "
        "so M1 is sourced but near the floor of measurability -- a result the "
        "Phase 23 test needs, not a gap.",
    ),
)


# ---------------------------------------------------------------------------
# Access / gate helpers.
# ---------------------------------------------------------------------------

def gate_record(record, *, allow_anchors=False):
    """Run one record's provenance species through the EXISTING species gate.

    Returns None when the record is human (it passes); raises
    `species_gate.UnsourcedParameterError` for any non-human species. This is
    the same `species_gate.gate` every human prediction already uses -- the gate
    is reused, not re-implemented, so a fingerprint value cannot enter a human
    comparison on a looser rule than a PK or endpoint parameter.
    """
    return species_gate.gate(
        record.species,
        layer="binding layer (off-target fingerprint)",
        label="%s Ki at %s (%s)" % (record.drug, record.target, record.reference),
        source="%s; %s" % (record.preparation, PDSP_SOURCE),
        allow_anchors=allow_anchors,
    )


def human_records(drug=None, target=None):
    """Human off-target Ki records, optionally filtered by drug and/or target.

    Every record returned passes `gate_record` by construction.
    """
    out = []
    for rec in HUMAN_KI:
        if drug is not None and rec.drug != drug:
            continue
        if target is not None and rec.target != target:
            continue
        out.append(rec)
    return tuple(out)


def human_ki_range_nM(drug, target):
    """`(min, max)` over the registered human values for one cell, in nM.

    The registry includes each cell's PDSP minimum and maximum by the selection
    rule, so this is the true full-set span, not a subset. Censored (">")
    ceilings are included at their bound value.
    """
    recs = human_records(drug, target)
    if not recs:
        raise KeyError("no human off-target Ki for %s at %s" % (drug, target))
    values = [r.value_nM for r in recs]
    return min(values), max(values)


def fingerprint(drug):
    """`{target: (min_nM, max_nM)}` human off-target fingerprint for one drug."""
    if drug not in DRUGS:
        raise KeyError("no off-target fingerprint for %r" % (drug,))
    return {t: human_ki_range_nM(drug, t) for t in TARGETS}


def cells_with_human_value():
    """The (drug, target) cells that resolved to at least one human value."""
    return {(r.drug, r.target) for r in HUMAN_KI}


def missing_cells():
    """(drug, target) cells in the DRUGS x TARGETS grid with NO human value.

    Empty by design: all ten cells are sourced. A non-empty return means a
    registered record was dropped and the fingerprint has a hole.
    """
    have = cells_with_human_value()
    return {(d, t) for d in DRUGS for t in TARGETS if (d, t) not in have}


__all__ = [
    "PDSP_RETRIEVED",
    "PDSP_SOURCE",
    "TARGETS",
    "DRUGS",
    "SIAFIS_2023",
    "OffTargetKi",
    "HUMAN_KI",
    "NON_HUMAN_EXCLUDED",
    "SEARCHED_AND_EMPTY",
    "gate_record",
    "human_records",
    "human_ki_range_nM",
    "fingerprint",
    "cells_with_human_value",
    "missing_cells",
]
