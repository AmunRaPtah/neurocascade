"""P11.2: the enforced species gate at the human boundary.

`docs/roadmap-human.md` moved neurocascade to human-primary and quarantined
the rodent layers under `legacy/`. Quarantine by import path is only half the
boundary: a rodent NUMBER can still walk into a human prediction through a
parameter set whose module was never moved, because almost every module here
mixes species -- a human affinity beside a rat association rate, a mouse Km
under a human Ki. P11.2 is the other half of the boundary:

    A PARAMETER WHOSE PROVENANCE SPECIES IS NOT HUMAN RAISES
    `UnsourcedParameterError` WHEN IT REACHES `fingerprint.predict()` OR ANY
    HUMAN SCENARIO BUILDER, UNLESS IT IS EXPLICITLY DECLARED A CROSS-SPECIES
    ANCHOR IN THE REGISTRY BELOW.

An anchor may be reported and compared; it may never drive a prediction. So
an anchored parameter is refused at every DRIVING position too, with its own
message saying so -- the registry licenses a non-human value to appear at the
human boundary in a REPORTING position (`allow_anchors=True`, used by the
anchor report and by any future comparison surface), never inside a
computation.

What carries provenance species, per layer
------------------------------------------

* **Receptor/binding layer** -- `BindingRecord`, defined here. The task
  brief describes the tree as already carrying these fields under this name;
  in fact no such structured field existed before P11.2 -- the receptor
  modules record each constant's species in comments and citation blocks
  only, which nothing programmatic can read. This module is that record,
  written by auditing every receptor module's own citations (the per-target
  tables below name the paper and preparation each constant comes from).
  `tests/test_species_gate.py` checks every record against the system its
  target actually builds, so the table cannot silently drift from the code.
* **PK layer** -- the PK models themselves. `pk.DosedOneCompartmentPK` and
  `pk.DosedTwoCompartmentPK` have carried `.species` since Phase 1;
  `pk.OneCompartmentPK` gained an optional one in this phase (default None,
  so existing illustrative uses are unchanged). The gate reads whatever the
  model declares.
* **Partition layer** -- `BrainPartition.species`, checked since P1.5 by
  `check_species`; the gate applies the same rule at the prediction boundary
  rather than only at attach time.
* **Endpoint layer** -- `EndpointTerm.species`, present since P7.1. The gate
  is `Endpoint.check_species`, wired into `Endpoint.resolve(target_species=
  ...)`: the human endpoint consumer arrives in P17 and must pass the
  species it is predicting for, and until then nothing can quietly resolve a
  mouse term onto a human readout by passing `target_species`.
* **Anatomy/expression layer** -- `legacy.expression.anatomy.AnatomySource
  .species`. Not wired here because it cannot reach a prediction: P11.1's
  test suite proves no module outside `legacy/` imports it.

Policy at the two edges, stated rather than left implicit
---------------------------------------------------------

* **A constant with NO species claim passes.** Three kinds carry no claim:
  the explicitly assumed or illustrative numbers (the benzodiazepine site's
  shared `kon`, `ILLUSTRATIVE_KON`), definitions and stated simplifications
  (efficacy 1.0 for a reference agonist), and MEASURED constants whose
  module states no preparation (Mann2022's opioid dissociation rates,
  Pertwee2010's compiled CB1 affinities). Refusing them would refuse
  parameters that are not known to be rodent, and labelling them human
  would be invention; they pass, each with the reason in its record, and
  `tests/test_species_gate.py::test_the_no_claim_set_is_pinned` freezes the
  exact set so it cannot grow unnoticed. Closing these gaps is P15's job.
* **A species string that is present but unknown REFUSES.** PDSP-style rows
  routinely say "RAT", "HEK293", "CLONED"; anything that does not normalize
  to a known species token cannot be asserted human, so it is refused with
  the raw value in the message and the caller decides. Absence of a claim
  and presence of an unreadable claim are different states and are treated
  differently on purpose.

The two seeded anchors
----------------------

Both come from `docs/roadmap-human.md`'s retirement policy, and both were
re-verified against the primary sources while seeding the registry:

* **TPA023 occupancy EC50**, Atack 2010 (PMID 19779131): the abstract states
  verbatim that EC50 for "rats, rhesus monkeys, and baboons were all similar
  and ranged from 19 to 30 ng/ml, although in humans, the EC(50) was
  slightly lower at 9 ng/ml".
* **Morphine Kp,uu**: the roadmap attributes BOTH values to Ederoth 2004
  (PMID 15025740). That paper supplies only the human value -- unbound
  brain:plasma AUC ratio 0.64 (95% CI 0.40-0.87) in 'better' tissue of seven
  traumatic-brain-injury patients. The rat value is from Tunblad 2003 (PMID
  12739770): steady-state unbound brain:blood ratio 0.29 +/- 0.07 in seven
  Sprague-Dawley rats (striatal microdialysis). The correction is recorded
  in docs/architecture.md; the registry cites each number to its own paper,
  and notes the blood-vs-plasma matrix difference between them.
"""

import re


class UnsourcedParameterError(LookupError):
    """Raised when a parameter whose provenance does not license the use it
    is being put to reaches a prediction or scenario builder.

    The canonical definition lives HERE since P11.2, because the species
    gate raises it across four layers and the gate module may not import
    `legacy.drug_pk` (nothing outside `legacy/` may). `legacy.drug_pk`
    re-imports it, so every pre-existing import path and except-clause keeps
    working, and a rodent PK lookup and a species-gate refusal are caught by
    the same clause -- both are "this parameter does not exist for this use".

    Two distinct situations raise it, and the messages distinguish them:

    * the parameter has been searched for and not found (drug_pk's original
      use -- see `SEARCHED_AND_EMPTY`);
    * the parameter EXISTS but was measured in a non-human species, so it
      may not enter a human prediction (the species gate's use).
    """


# ---------------------------------------------------------------------------
# Species tokens
# ---------------------------------------------------------------------------

#: The canonical token for the modelling species.
HUMAN = "human"

#: Recognized spellings -> canonical token. Deliberately SHORT: a name is
#: mapped only when it names the animal unambiguously. Cell-line names
#: ("HEK293", "CHO"), preparation words ("CLONED") and strain names are NOT
#: here -- they do not say which species' receptor was measured, which is
#: exactly the fact the gate exists to check, so they stay unrecognized and
#: refuse.
SPECIES_SYNONYMS = {
    "human": (
        "human", "homo sapiens", "homo sapien", "human receptor",
        "humans",
    ),
    "mouse": ("mouse", "mus musculus", "mice"),
    "rat": ("rat", "rattus norvegicus", "rats"),
    "rhesus": ("rhesus", "rhesus macaque", "rhesus monkey",
               "macaca mulatta"),
    "baboon": ("baboon",),
}

_REVERSED_SPECIES = {
    spelling: canonical
    for canonical, spellings in SPECIES_SYNONYMS.items()
    for spelling in spellings
}


def normalize_species(value):
    """The canonical species token for `value`, or None when there is none.

    None means NO CLAIM was made -- the argument was None or empty -- which
    the gate treats as "no provenance to check" and lets through. Anything
    else that does not map is returned lowercased and stripped, so it can
    never compare equal to `HUMAN` and will surface verbatim in the refusal.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    mapped = _REVERSED_SPECIES.get(text)
    if mapped is not None:
        return mapped
    # Tolerate decorations around a recognized name: "rat (probable)",
    # "human recombinant", "Mus musculus (C57BL/6J)".
    for spelling, canonical in _REVERSED_SPECIES.items():
        if re.search(r"\b%s\b" % re.escape(spelling), text):
            return canonical
    return text


# ---------------------------------------------------------------------------
# The binding-layer provenance record
# ---------------------------------------------------------------------------


class SourceMeasurement(object):
    """One literature measurement (or explicit non-measurement) behind one
    binding-layer quantity.

    quantity : str
        What the number is, e.g. "Ki", "kon (shared)", "EC50 used as Kd",
        "k_des/k_rec".
    species : str or None
        The species the measurement was made in, normalized by
        `normalize_species`. REQUIRED for a literature measurement; None
        means the quantity makes no species claim at all, and `note` says
        why (assumed, illustrative, definition, or measured-but-unstated).
    source : str
        Citation key plus the preparation, as the source module records it.
    """

    __slots__ = ("quantity", "species", "source", "note")

    def __init__(self, quantity, species, source, note=""):
        self.quantity = str(quantity)
        self.species = normalize_species(species)
        self.source = str(source)
        self.note = str(note)

    @property
    def is_non_human(self):
        return self.species is not None and self.species != HUMAN

    def __repr__(self):
        return "SourceMeasurement(%r, species=%r, %s)" % (
            self.quantity, self.species, self.source
        )


class BindingRecord(object):
    """One modelled binding site's literature-species provenance.

    `ligands` maps each ligand name at the site to the measurements behind
    its constants; `site_constants` holds the quantities that belong to the
    site rather than to one ligand (a shared desensitization rate pair, a
    synaptic-share input). Everything here is transcribed from the source
    modules' own citation blocks; the completeness test in
    `tests/test_species_gate.py` builds each target's system and checks the
    two against each other, so a ligand added to a site without a record --
    or a record for a ligand no longer at the site -- fails the suite.
    """

    def __init__(self, target, site_name, ligands=None, site_constants=None,
                 note=""):
        self.target = str(target)
        self.site_name = str(site_name)
        self.ligands = {
            name: tuple(measurements)
            for name, measurements in dict(ligands or {}).items()
        }
        self.site_constants = tuple(site_constants or ())
        self.note = str(note)

    @property
    def measurements(self):
        for name in sorted(self.ligands):
            for measurement in self.ligands[name]:
                yield name, measurement
        for measurement in self.site_constants:
            yield None, measurement

    def non_human(self):
        """`(ligand_or_None, measurement)` for every non-human measurement."""
        return [
            (name, measurement)
            for name, measurement in self.measurements
            if measurement.is_non_human
        ]

    @property
    def is_human_provenanced(self):
        return not self.non_human()

    def __repr__(self):
        return "BindingRecord(%r, %r, %d non-human measurement(s))" % (
            self.target, self.site_name, len(self.non_human())
        )


#: Every modelled fingerprint target -> its record. Keyed by canonical
#: TARGETS name; `SITE_BINDING_RECORDS` repeats the mapping by site name for
#: callers holding a site rather than a target (the human scenario builder).
BINDING_RECORDS = {}
SITE_BINDING_RECORDS = {}


def _register_record(record):
    if record.target in BINDING_RECORDS:
        raise ValueError("duplicate BindingRecord for %r" % record.target)
    BINDING_RECORDS[record.target] = record
    SITE_BINDING_RECORDS[record.site_name] = record
    return record


def _m(quantity, species, source, note=""):
    return SourceMeasurement(quantity, species, source, note)


# --- opioid receptors -------------------------------------------------------

_register_record(BindingRecord(
    target="MOR",
    site_name="mu_opioid_on_VTA_GABA",
    ligands={
        "fentanyl": (
            _m("Ki", "human",
               "[Volpe2011] single standardized [3H]-DAMGO competition assay "
               "on recombinant human MOR membranes"),
            _m("koff", None,
               "[Mann2022] in vitro binding-kinetics fit",
               note="measured; neither scenarios.py nor opioid.py states the "
                    "preparation's species, so no claim is made"),
        ),
        "naloxone": (
            _m("Ki", "human",
               "[Volpe2011] same assay, reference ligand"),
            _m("koff", None,
               "[Mann2022] in vitro binding-kinetics fit",
               note="species not stated; see fentanyl"),
        ),
    },
    note=(
        "kon is derived (koff/Ki) for both ligands and inherits the union of "
        "its inputs' provenances. The unstated koff species is the gap P16.2 "
        "(the FDA receptor-to-ventilation port) works against."
    ),
))

_OPIOID_LIGANDS = (
    "dynorphin_A", "leu_enkephalin", "U50488", "nor_BNI", "naltrindole",
    "fentanyl", "naloxone",
)


def _toll1998(subtype, censored_at_kappa=False):
    """`[Toll1998]`'s Ki/Emax measurement at one opioid subtype, plus the
    censored-cell caveat where the paper printed `>10000` instead of a
    number (U50488 at delta, Leu-enkephalin at kappa)."""
    measurements = [_m(
        "Ki (+ Emax)", "human",
        "[Toll1998] Tables 5 & 7, human cloned %s receptor in CHO cells"
        % subtype,
    )]
    if censored_at_kappa:
        measurements.append(_m(
            "this ligand's Ki is a censored >10000 nM assay ceiling, "
            "not a measurement", None,
            "[Toll1998] assay ceiling (opioid.ASSAY_CEILING_NM)",
            note="a bound carried at the same human-CHO assay it comes from",
        ))
    return tuple(measurements)


_register_record(BindingRecord(
    target="DOR",
    site_name="delta_opioid_on_VTA_GABA",
    ligands={
        name: _toll1998("delta")
        for name in _OPIOID_LIGANDS
    },
    site_constants=(
        _m("kon (shared radioligand)", "rat",
           "[Vaughn1989] [3H]pCl-DPDPE slow component, rat brain homogenate"),
    ),
))

_register_record(BindingRecord(
    target="KOR",
    site_name="kappa_opioid_on_mesolimbic_DA_terminal",
    ligands={
        name: _toll1998("kappa", censored_at_kappa=(name == "leu_enkephalin"))
        for name in _OPIOID_LIGANDS
    },
    site_constants=(
        _m("kon (shared radioligand)", "human",
           "[Wang2021] [3H]diprenorphine competition association, human "
           "KOR-expressing cell membranes"),
    ),
))


# --- dopamine ---------------------------------------------------------------

_register_record(BindingRecord(
    target="D1",
    site_name="D1_on_striatal_direct_MSN",
    ligands={
        "dopamine": (
            _m("kon + koff", "human",
               "[Sahlholm2025] human D1 + GIRK1/4 in Xenopus oocytes, GIRK "
               "current kinetics"),
            _m("efficacy 1.0", None, "reference agonist by construction",
               note="definition, no measurement claimed"),
        ),
        "dihydrexidine": (
            _m("kon + koff + Emax", "human",
               "[Sahlholm2025] same assay and table"),
        ),
        "SCH-23390": (
            _m("kon (shared from dopamine)", "human",
               "[Sahlholm2025] derivation 3"),
            _m("Kd", "rat",
               "[Billard1984] [3H]SCH-23390 saturation binding, rat striatal "
               "membranes"),
            _m("efficacy 0.0", None, "modelling simplification",
               note="stated simplification, no measurement claimed"),
        ),
    },
))

_register_record(BindingRecord(
    target="D2",
    site_name="D2_on_striatal_indirect_MSN",
    ligands={
        "dopamine": (
            _m("kon + koff", "human",
               "[Agren2021] human D2 + GIRK1/4 + RGS4 in Xenopus oocytes"),
            _m("efficacy 1.0", None, "reference agonist by construction",
               note="definition, no measurement claimed"),
        ),
        "haloperidol": (
            _m("kon + koff", "human",
               "[Sykes2017] TR-FRET competition kinetics at human D2L "
               "receptors"),
            _m("efficacy 0.0", None, "neutral antagonist",
               note="stated simplification, no measurement claimed"),
        ),
    },
))


# --- serotonin --------------------------------------------------------------

_register_record(BindingRecord(
    target="5HT1A",
    site_name="5HT1A_raphe_and_cortex",
    ligands={
        "serotonin": (
            _m("kon + Ki", "human",
               "[Satala2016] human 5-HT1A stably expressed in HEK293 cells"),
        ),
        "8-OH-DPAT": (
            _m("kon + Kd", "human", "[Satala2016] same assay"),
            _m("efficacy", "rat",
               "[NewmanTancredi2003] [35S]GTPgammaS panel, rat hippocampal "
               "membranes"),
        ),
        "WAY-100635": (
            _m("koff + Kd", "rat",
               "[Parkel2006] [3H]WAY-100635 saturation and dissociation, rat "
               "hippocampal membranes"),
            _m("efficacy 0.0 (measured)", "rat",
               "[NewmanTancredi2003]/[Forster1995]/[Fletcher1996] rat "
               "preparations"),
        ),
    },
))

_register_record(BindingRecord(
    target="5HT2A",
    site_name="5HT2A_on_cortical_pyramidal",
    note=(
        "Provenances below are for the DEFAULT live-cell affinity state; the "
        "antagonist-radioligand state swaps serotonin's affinity to "
        "[McKenna1989] rat cortex."
    ),
    ligands={
        "serotonin": (
            _m("kon + koff", "human",
               "[Denomme2024] NanoBRET kinetics at human 5-HT2A in live "
               "HEK293 cells"),
        ),
        "DOI": (
            _m("kon (shared)", "human", "[Denomme2024] derivation 3"),
            _m("Ki", "rat",
               "[Seggel1990 via Hemanth2023] rat cortical [3H]ketanserin-"
               "labelled 5-HT2 sites"),
            _m("efficacy", "human",
               "[Cussac2008] h5-HT2A Gq/11 calcium mobilisation in CHO cells"),
        ),
        "ketanserin": (
            _m("kon (shared)", "human", "[Denomme2024] derivation 3"),
            _m("Kd", "rat",
               "[McKenna1989] [3H]ketanserin saturation, rat cortex"),
            _m("efficacy 0.0", None, "modelling simplification",
               note="stated simplification, no measurement claimed"),
        ),
    },
))


# --- noradrenaline ----------------------------------------------------------

_register_record(BindingRecord(
    target="alpha1",
    site_name="alpha1_on_cortical_pyramidal",
    ligands={
        "noradrenaline": (
            _m("kon (shared from prazosin)", "rat",
               "[Tanaka2004] [3H]prazosin association, intact rat tail artery "
               "segments at 4 C"),
            _m("Kd", "human",
               "[Proudman2021] [3H]prazosin whole-cell binding, human "
               "alpha1A in CHO cells"),
        ),
        "prazosin": (
            _m("kon", "rat",
               "[Tanaka2004] intact rat tail artery segments at 4 C"),
            _m("koff", "rat",
               "[Quinton2010] [3H]prazosin dissociation, rat brain "
               "synaptosomes"),
            _m("efficacy 0.0", None, "modelling simplification",
               note="stated simplification, no measurement claimed"),
        ),
    },
))

_register_record(BindingRecord(
    target="alpha2",
    site_name="alpha2_autoreceptor_on_LC",
    note=(
        "Provenances below are for the DEFAULT agonist-state; the whole-cell "
        "state swaps noradrenaline's and clonidine's affinities to "
        "[Proudman2022] human alpha2A CHO whole-cell values."
    ),
    ligands={
        "noradrenaline": (
            _m("kon (shared)", "human",
               "[Gerhardt1990] p-[125I]iodoclonidine association, human "
               "platelet membranes"),
            _m("Kd (agonist state)", "rat",
               "[UPrichard1978] (-)-[3H]noradrenaline saturation, rat cortex "
               "membranes"),
        ),
        "clonidine": (
            _m("kon (shared)", "human", "[Gerhardt1990] derivation 3"),
            _m("Kd (agonist state)", "rat",
               "[Salama1982] [3H]clonidine high-affinity site, rat cortex "
               "membranes"),
            _m("efficacy", "human",
               "[Jasper1998] cloned human alpha2a [35S]GTPgammaS"),
        ),
        "yohimbine": (
            _m("kon (shared)", "human", "[Gerhardt1990] derivation 3"),
            _m("Kd", "human",
               "[Proudman2022b] [3H]rauwolscine whole-cell binding, human "
               "alpha2A in CHO cells"),
            _m("efficacy 0.0 (measured)", "human",
               "[Jasper1998]/[Millan2000], human alpha2"),
        ),
    },
))

_register_record(BindingRecord(
    target="beta",
    site_name="beta1_on_cortical_pyramidal",
    ligands={
        "noradrenaline": (
            _m("kon (isoprenaline's, shared)", "human",
               "[Sykes2022] competition association at human beta1 in CHO "
               "membranes"),
            _m("Kd", "human",
               "[Baker2010] [3H]CGP 12177 whole-cell binding, human beta "
               "subtypes in CHO cells"),
            _m("efficacy", "human",
               "[Baker2010] 102.4% of isoprenaline Emax, human beta1 cAMP"),
        ),
        "propranolol": (
            _m("kon + koff", "human",
               "[Sykes2022] human beta1 CHO membrane competition kinetics"),
            _m("efficacy 0.0", None, "modelling simplification",
               note="stated simplification, no measurement claimed"),
        ),
    },
))


# --- acetylcholine ----------------------------------------------------------

_CHOLINERGIC_NO_CLAIM = _m(
    "kon", None,
    "acetylcholine.ILLUSTRATIVE_KON, shared by all four cholinergic sites",
    note="illustrative: no measured kon exists for any ligand at any of "
         "these sites",
)

_register_record(BindingRecord(
    target="alpha4beta2",
    site_name="alpha4beta2_on_cortical_pyramidal",
    ligands={
        "acetylcholine": (
            _m("EC50 used as conducting-state Kd", "human",
               "[Sanders2022] high-sensitivity human (alpha4)2(beta2)3 "
               "concatemer in Xenopus oocytes"),
        ),
        "nicotine": (
            _m("EC50 used as Kd", "rat",
               "[Fenster1997] rat alpha4beta2 in Xenopus oocytes"),
        ),
        "mecamylamine": (
            _m("IC50", None,
               "[Papke2013] peak-current IC50 at high-sensitivity "
               "alpha4beta2; the module states the subunit species nowhere",
               note="measured; preparation's species not stated"),
        ),
    },
    site_constants=(
        _m("k_des/k_rec", "rat",
           "[Fenster1997] Table 1 desensitization/recovery time constants, "
           "rat alpha4beta2"),
        _CHOLINERGIC_NO_CLAIM,
    ),
))

_register_record(BindingRecord(
    target="alpha7",
    site_name="alpha7_on_cortical_pyramidal",
    ligands={
        "acetylcholine": (
            _m("EC50 used as Kd", "rat",
               "[PapkePapke2002] net-charge analysis, rat alpha7 in Xenopus "
               "oocytes"),
        ),
        "nicotine": (
            _m("EC50 used as Kd", "rat",
               "[Fenster1997] rat alpha7 in Xenopus oocytes"),
        ),
        "mecamylamine": (
            _m("IC50 (~50% block at 3 uM)", "human",
               "[ChavezNoriega1997] recombinant human alpha7 in Xenopus "
               "oocytes"),
        ),
    },
    site_constants=(
        _m("k_des/k_rec", "rat",
           "[Fenster1997] Table 1 time constants, rat alpha7"),
        _CHOLINERGIC_NO_CLAIM,
    ),
))

for _subtype, _ach_kd_note in (
    ("M1", "midpoint of curated pKi range 4.3-4.9"),
    ("M2", "midpoint of curated pKi range 4.3-6.5"),
):
    _register_record(BindingRecord(
        target=_subtype,
        site_name="%s_%s" % (
            _subtype,
            "autoreceptor_on_basal_forebrain" if _subtype == "M2"
            else "on_cortical_pyramidal",
        ),
        ligands={
            "acetylcholine": (
                _m("Kd (%s)" % _ach_kd_note, "human",
                   "[GtoPdb2026] Guide to PHARMACOLOGY, human pKi as curated"),
            ),
            "scopolamine": (
                _m("Kd", "human",
                   "[Bolden1992]/[Huang2001] cloned human muscarinic receptors"
                   " in CHO cells, read via the Guide's transcription"),
                _m("efficacy 0.0", None, "modelling simplification",
                   note="stated simplification, no measurement claimed"),
            ),
        },
        site_constants=(_CHOLINERGIC_NO_CLAIM,),
    ))

del _subtype, _ach_kd_note


# --- glutamate --------------------------------------------------------------

_NMDA_SHARE_CONSTANTS = (
    _m("synaptic share input: NMDA:AMPA peak-current ratio", "rat",
       "[Myme2003] layer 2/3 pyramidal cells, rat medial prefrontal and "
       "visual cortex"),
    _m("synaptic share input: Mg-block conductance-voltage fit", "mouse",
       "[ChiuCarter2022] acute mouse cortical slices"),
)

_register_record(BindingRecord(
    target="NMDA",
    site_name="NMDA_on_glutamatergic_connection",
    ligands={
        "glutamate": (
            _m("EC50 used as Kd", "mouse",
               "[Patneau1990] mouse embryonic hippocampal neurons in culture"),
            _m("kon", None, "glutamate.ILLUSTRATIVE_KON",
               note="illustrative: no measured orthosteric kon exists"),
        ),
        "ketamine": (
            _m("IC50 used as channel-pool Kd", "rat",
               "[Kotermanski2009] rat GluN1/GluN2A in HEK293T cells"),
            _m("kon (MK-801's, shared within the pool)", "rat",
               "[Dravid2007] rat NR1/NR2A in Xenopus oocytes"),
        ),
        "mk801": (
            _m("kon + koff", "rat",
               "[Dravid2007] rat NR1/NR2A in Xenopus oocytes"),
        ),
    },
    site_constants=_NMDA_SHARE_CONSTANTS,
))

_register_record(BindingRecord(
    target="AMPA",
    site_name="AMPA_on_glutamatergic_connection",
    ligands={
        "glutamate": (
            _m("EC50 used as Kd (rapidly desensitizing)", "mouse",
               "[Patneau1990] mouse embryonic hippocampal neurons in culture"),
            _m("kon", None, "glutamate.ILLUSTRATIVE_KON",
               note="illustrative: no measured orthosteric kon exists"),
        ),
        "nbqx": (
            _m("apparent Ki", "rat",
               "[Randle1992] rat cortex mRNA expressed in Xenopus oocytes"),
            _m("kon", None, "glutamate.ILLUSTRATIVE_KON",
               note="illustrative: no measured orthosteric kon exists"),
        ),
    },
    site_constants=_NMDA_SHARE_CONSTANTS,
))


# --- histamine, cannabinoid, adenosine --------------------------------------

_register_record(BindingRecord(
    target="H1",
    site_name="H1_on_cortical_pyramidal",
    ligands={
        "histamine": (
            _m("Ki", "human",
               "[Biselli2021] human H1 expressed in Sf9 insect cells, "
               "[3H]mepyramine displacement"),
        ),
        "diphenhydramine": (
            _m("koff (from residence time)", "human",
               "[Bosma2018] residence time at the human H1 receptor"),
            _m("Ki", "human",
               "[Booth2002] cloned human H1 in CHO membranes"),
        ),
        "mepyramine": (
            _m("kon + koff", "human",
               "[Bosma2017] competition association, human H1 in HEK293T "
               "membranes"),
        ),
    },
))

_register_record(BindingRecord(
    target="CB1",
    site_name="CB1_on_GABAergic_terminal",
    ligands={
        "2ag": (
            _m("Ki bounds", None,
               "[Pertwee2010] Table 1 compiled ranges; the module attributes "
               "no species or preparation to any CB1 affinity",
               note="measured; species not stated in the compilation"),
            _m("efficacy 1.0 (reference standing)", "rat",
               "[Savinainen2001] full-efficacy agonism, rat cerebellar "
               "membranes"),
        ),
        "anandamide": (
            _m("Ki bounds", None, "[Pertwee2010] Table 1; species not stated",
               note="measured; species not stated in the compilation"),
            _m("efficacy 0.61", "rat",
               "rat CB1 in HEK293 cells, cAMP inhibition"),
        ),
        "thc": (
            _m("Ki bounds", None, "[Pertwee2010] Table 1; species not stated",
               note="measured; species not stated in the compilation"),
            _m("efficacy 0.20", "human",
               "human CB1 in HEK293-EBNA cells, [35S]GTPgammaS binding"),
        ),
        "rimonabant": (
            _m("Ki bounds", None, "[Pertwee2010] Table 1; species not stated",
               note="measured; species not stated in the compilation"),
            _m("efficacy 0.0", "human",
               "[human CB1 CHO membranes: antagonist decreases GTPgammaS "
               "binding below basal]"),
        ),
    },
    site_constants=(
        _m("kon", None, "cannabinoid.ILLUSTRATIVE_KON",
           note="illustrative: no CB1 binding kinetics were found"),
    ),
))

_register_record(BindingRecord(
    target="A2A",
    site_name="A2A_on_striatopallidal_MSN",
    note=(
        "adenosine.receptor_system_a2a defaults species='rat'; the module "
        "also defines the human set ([Fredholm2011] adenosine 310 nM, "
        "caffeine 23400 nM), and P15.1 flips the default."
    ),
    ligands={
        "adenosine": (
            _m("functional potency used as Kd", "rat",
               "[Fredholm2011] Table 3, rat column"),
        ),
        "caffeine": (
            _m("Kd", "rat", "[Fredholm2011] Table 4, rat column"),
        ),
    },
    site_constants=(
        _m("kon (NECA's, shared)", "human",
           "[Guo2012] competition association at human A2A in HEK293 "
           "membranes"),
    ),
))


# --- the GABA-A benzodiazepine site ------------------------------------------

_register_record(BindingRecord(
    target="GABA_A_BZD",
    site_name="GABA_A_benzodiazepine_site_on_BLA_interneuron_synapse",
    ligands={
        "diazepam": (
            _m("Ki (= Kd 3.6 nM)", "rat",
               "[Mohler1977] [3H]diazepam saturation binding, rat cerebral "
               "cortex synaptosomal membranes"),
        ),
        "flumazenil": (
            _m("Ki (= Kd 1.0 nM)", "rat",
               "[Mohler1981] [3H]Ro 15-1788 saturation binding, same rat "
               "cerebral cortex preparation"),
        ),
    },
    site_constants=(
        _m("kon_shared = 0.2", None,
           "scenarios.gaba_a_benzodiazepine_ligands_calibrated",
           note="assumed, not measured; chosen to set a timescale"),
    ),
))


# --- monoamine transporters --------------------------------------------------

_PSYCHOSTIMULANT_KI = (
    _m("Ki", "mouse",
       "[HanGu2006] Table 1, MOUSE transporter column, uptake inhibition in "
       "transfected Intestine 407 cells"),
)
_ANTIDEPRESSANT_KI = (
    _m("Ki", "human",
       "atomoxetine [Bymaster2002]; fluoxetine [Tatsumi1997] via IUPHAR "
       "curation; human transporters, radioligand binding"),
)
_AMPHETAMINE_RELEASE = (
    _m("release EC50 + Emax", "rat",
       "[Rothman2012] Table 1, d-amphetamine, rat brain synaptosomes"),
)


def _transporter_record(transporter, uptake, basal):
    """One transporter's record over the drugs `drugs_at()` keeps by default."""
    ligands = {
        "cocaine": _PSYCHOSTIMULANT_KI,
        "methylphenidate": _PSYCHOSTIMULANT_KI,
        "amphetamine": _PSYCHOSTIMULANT_KI + _AMPHETAMINE_RELEASE,
        "atomoxetine": _ANTIDEPRESSANT_KI,
    }
    if transporter == "SERT":
        ligands["fluoxetine"] = _ANTIDEPRESSANT_KI
    return BindingRecord(
        target=transporter,
        site_name=transporter,
        ligands=ligands,
        site_constants=(uptake, basal),
        note=(
            "PK at every transporter drug is monoamine_transporters."
            "TOOL_COMPOUND_KE, ILLUSTRATIVE, so it carries no species claim."
        ),
    )


_register_record(_transporter_record(
    "DAT",
    _m("Km/Vmax of dopamine uptake", "mouse",
       "[JohnJones2007] mouse caudate-putamen slices, FSCV"),
    _m("basal extracellular dopamine", "mouse",
       "[Bosse2011] zero-net-flux microdialysis, wild-type mouse "
       "caudate-putamen"),
))
_register_record(_transporter_record(
    "SERT",
    _m("Km/Vmax of serotonin uptake", "mouse",
       "[JohnJones2007] mouse substantia nigra pars reticulata slices"),
    _m("basal extracellular serotonin", "mouse",
       "[Mathews2004] zero-net-flux microdialysis, wild-type mouse striatum"),
))
_register_record(_transporter_record(
    "NET",
    _m("noradrenaline clearance (first-order form)", "rat",
       "[Park2012] FSCV clearance half-life, rat ventral BNST"),
    _m("basal extracellular noradrenaline", "rat",
       "[PanLai1995] zero-net-flux microdialysis, conscious rat medial "
       "prefrontal cortex"),
))


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------


def _refuse(layer, label, species, source, anchor=None, role=None):
    raise UnsourcedParameterError(
        "%s: %s carries provenance species %r, which is not %r, so it cannot "
        "%s. %s%s"
        % (
            layer,
            label,
            species,
            HUMAN,
            "be reported in a human comparison"
            if role == "report"
            else "enter a human prediction",
            "Source of the parameter in hand: %s. " % source if source else "",
            _anchor_clause(anchor),
        )
    )


def _anchor_clause(anchor):
    if anchor is None:
        return ""
    return (
        "This parameter IS a registered cross-species calibration anchor "
        "(%s): anchors exist to quantify the transfer error between species "
        "and may be reported and compared, but they may never drive a "
        "prediction. See species_gate.CROSS_SPECIES_ANCHORS[%r]."
        % (anchor.parameter, anchor.key)
    )


def gate(species, *, layer, label, source="", allow_anchors=False,
         anchor_key=None):
    """The one gate every layer goes through.

    `species` is the parameter's provenance species, in any spelling
    `normalize_species` reads, or None for no claim. `layer` and `label`
    name the parameter in the refusal ("partition layer: morphine/rat
    Kp,uu"). `source` is the citation of the parameter in hand, quoted in
    the refusal.

    `allow_anchors=False` (every prediction and scenario builder passes
    this) refuses EVERY non-human species, anchored or not; the message for
    an anchored parameter explains that anchors never drive predictions.
    `allow_anchors=True` is the reporting mode: a non-human species covered
    by a registered anchor returns that anchor instead of raising, so a
    comparison surface can show it next to its human counterpart. An
    anchored species is found whether or not `anchor_key` narrows it; a
    non-human species NO anchor covers always raises, in both modes.
    """
    normalized = normalize_species(species)
    if normalized is None or normalized == HUMAN:
        return None

    anchor = declared_anchor(normalized, key=anchor_key)
    if anchor is not None and allow_anchors:
        return anchor
    _refuse(layer, label, normalized, source, anchor=anchor,
            role="report" if allow_anchors else "input")


# --- per-layer entry points -------------------------------------------------
#
# Each wraps `gate()` with the layer's own way of carrying provenance.


def check_pk_model(model, *, allow_anchors=False):
    """`gate()` over a PK model's own declared species.

    Models that declare no species (`OneCompartmentPK` built without one,
    foreign duck-typed models) pass: the gate refuses KNOWN rodent
    provenance, and a model with no species attribute makes no claim. The
    attached partition, if any, is checked separately, because its species
    belongs to a different measurement.
    """
    species = getattr(model, "species", None)
    gate(
        species,
        layer="PK layer",
        label="PK model %r" % getattr(model, "name", model),
        source="",
        allow_anchors=allow_anchors,
    )
    partition = getattr(model, "brain_partition", None)
    if partition is not None:
        check_brain_partition(partition, allow_anchors=allow_anchors)
    return model


def check_brain_partition(partition, *, allow_anchors=False):
    """`gate()` over a `BrainPartition`'s recorded species."""
    return gate(
        partition.species,
        layer="partition layer",
        label="%s/%s (%s=%g)"
        % (partition.drug, partition.species, partition.basis, partition.ratio),
        source=partition.source,
        allow_anchors=allow_anchors,
        # The two seeded anchors are both partition-shaped facts (morphine's
        # literally is a Kp,uu; TPA023's is the occupancy EC50 a partition
        # feeds), so anchor lookups are keyed by drug name where one fits.
        anchor_key=ANCHOR_KEY_BY_DRUG.get(str(partition.drug).lower()),
    )


def check_binding_target(target, entry=None, *, allow_anchors=False):
    """`gate()` over one fingerprint target's binding-layer provenance.

    Checks the site's own `BindingRecord` -- every ligand already at the
    site joins the competitive pool a prediction computes over -- plus, when
    given, the fingerprint ENTRY for the predicted compound itself, whose
    `species` is the row-level provenance P8.1 already carries.
    """
    from . import fingerprint as _fp

    canonical = _fp.resolve_target(target)
    spec = _fp.TARGETS[canonical]
    record = BINDING_RECORDS.get(canonical)
    if record is None:
        raise UnsourcedParameterError(
            "binding layer: target %r has no BindingRecord, so its "
            "provenance species cannot be checked and it cannot enter a "
            "human prediction. Register one in species_gate.BINDING_RECORDS."
            % canonical
        )
    offenders = record.non_human()
    if offenders:
        name, measurement = offenders[0]
        _refuse(
            "binding layer",
            "%s at %s: %s (%s)"
            % (
                name or "site constant",
                record.site_name,
                measurement.quantity,
                ", ".join(
                    sorted({m.species for _, m in offenders})
                ),
            ),
            measurement.species,
            measurement.source,
        )
    if entry is not None and getattr(entry, "species", None) is not None:
        gate(
            entry.species,
            layer="binding layer",
            label="%s Ki row for %s"
            % (canonical, getattr(entry, "target", canonical)),
            source=getattr(entry, "reference", "") or "",
            allow_anchors=allow_anchors,
        )
    return record


def check_receptor_site(site, *, allow_anchors=False):
    """`gate()` over a hand-built site reaching a human scenario builder.

    Looks the site up by NAME in `SITE_BINDING_RECORDS`; a site with no
    record raises, because an unchecked site is exactly how a rodent number
    walks past this gate.
    """
    record = SITE_BINDING_RECORDS.get(site.name)
    if record is None:
        raise UnsourcedParameterError(
            "binding layer: receptor site %r has no BindingRecord, so its "
            "constants' provenance species cannot be checked and it cannot "
            "be wired into a human scenario. Register one under "
            "species_gate.BINDING_RECORDS." % (site.name,)
        )
    offenders = record.non_human()
    if offenders:
        name, measurement = offenders[0]
        _refuse(
            "binding layer",
            "%s at %s: %s (%s)"
            % (
                name or "site constant",
                record.site_name,
                measurement.quantity,
                ", ".join(sorted({m.species for _, m in offenders})),
            ),
            measurement.species,
            measurement.source,
        )
    return record


def check_endpoint(endpoint, target_species, *, allow_anchors=False):
    """`gate()` over every term of an `Endpoint`.

    Endpoint terms are region-behind-a-behaviour claims taken from named
    experiments in named species; a term from a mouse lesion has no business
    driving a human behavioural readout, and `endpoints.Endpoint.resolve`
    calls this whenever the caller declares the species being predicted for.
    """
    offenders = []
    for term in endpoint.terms:
        normalized = normalize_species(term.species)
        if normalized is not None and normalized != HUMAN:
            offenders.append(term)
    if not offenders:
        return None
    species_list = sorted({normalize_species(t.species) for t in offenders})
    anchors = [declared_anchor(s) for s in species_list]
    if allow_anchors and all(anchors):
        return anchors
    citations = ", ".join(sorted({t.citation for t in offenders}))
    _refuse(
        "endpoint layer",
        "%d of %d terms of %r (species %s; citations: %s)"
        % (
            len(offenders),
            len(endpoint.terms),
            endpoint.key,
            ", ".join(species_list),
            citations[:140] + ("..." if len(citations) > 140 else ""),
        ),
        species_list[0],
        "endpoints.CITATIONS for the terms named",
    )


# ---------------------------------------------------------------------------
# Cross-species calibration anchors
# ---------------------------------------------------------------------------


class AnchorMeasurement(object):
    """One species' side of a cross-species anchor pair.

    Exactly one of `value` or `interval` carries the number: a point value
    when the paper reports one, an interval when it reports a range (or a
    confidence interval, which is kept apart from a point estimate rather
    than collapsed to its midpoint -- collapsing would be inventing a
    precision the paper did not claim).
    """

    __slots__ = ("species", "value", "interval", "unit", "source", "note")

    def __init__(self, species, unit, source, value=None, interval=None,
                 note=""):
        if (value is None) == (interval is None):
            raise ValueError(
                "an anchor measurement carries exactly one of a point value "
                "or an interval; got value=%r interval=%r" % (value, interval)
            )
        self.species = normalize_species(species)
        if self.species is None:
            raise ValueError("an anchor measurement needs a species")
        self.value = None if value is None else float(value)
        self.interval = (
            None if interval is None else (float(interval[0]), float(interval[1]))
        )
        self.unit = str(unit)
        self.source = str(source)
        self.note = str(note)

    def describe(self):
        number = (
            "%g" % self.value if self.value is not None
            else "%g-%g" % self.interval
        )
        bits = ["%s: %s %s" % (self.species, number, self.unit)]
        if self.note:
            bits.append("(%s)" % self.note)
        bits.append("[%s]" % self.source)
        return " ".join(bits)


class CrossSpeciesAnchor(object):
    """A declared cross-species calibration anchor.

    An anchor is a PAIR of measurements of the SAME parameter in different
    species, at least one of them non-human. It quantifies the transfer
    error the pivot accepted when it retired the rodent layers, and that is
    all it is good for: `may_drive_predictions` is structurally False, the
    gate refuses an anchored species at every driving position, and the
    reporting mode exists so the pair can be shown and compared.
    """

    def __init__(self, key, parameter, measurements, note=""):
        self.key = str(key)
        self.parameter = str(parameter)
        self.measurements = tuple(measurements)
        self.note = str(note)
        humans = [m for m in self.measurements if m.species == HUMAN]
        if len(humans) != 1:
            raise ValueError(
                "an anchor needs exactly one human measurement to compare "
                "against; %r has %d" % (key, len(humans))
            )
        if len([m for m in self.measurements if m.species != HUMAN]) < 1:
            raise ValueError(
                "an anchor needs at least one non-human measurement; %r has "
                "none, so there is nothing for it to anchor" % key
            )

    @property
    def human(self):
        return next(m for m in self.measurements if m.species == HUMAN)

    @property
    def non_human(self):
        return tuple(m for m in self.measurements if m.species != HUMAN)

    def covers(self, species):
        return normalize_species(species) in {
            m.species for m in self.measurements
        }

    #: Structural, not a policy someone could switch off: an anchor is a
    #: comparison object, and there is no code path that turns one into a
    #: prediction input.
    may_drive_predictions = False

    def transfer_ratios(self):
        """`{species: non_human/human}` per point-valued measurement.

        Only point estimates produce a ratio; an interval reports its span
        instead, because dividing a range by a point is a range and printing
        one endpoint would look like a measurement.
        """
        if self.human.value is None:
            # A human CI cannot anchor a ratio; describing() already shows
            # both sides' numbers, so report nothing rather than divide by a
            # range's midpoint we were not given.
            return {}
        ratios = {}
        for measurement in self.non_human:
            if measurement.value is not None:
                ratios[measurement.species] = (
                    measurement.value / self.human.value
                )
            else:
                lo, hi = measurement.interval
                ratios[measurement.species] = "%.4g-%.4g (range)" % (
                    lo / self.human.value, hi / self.human.value,
                )
        return ratios

    def describe(self):
        lines = ["%s -- %s" % (self.key, self.parameter)]
        for measurement in self.measurements:
            lines.append("  %s" % measurement.describe())
        lines.append("  drives predictions: %s" % self.may_drive_predictions)
        if self.note:
            lines.append("  note: %s" % self.note)
        return "\n".join(lines)


#: The registry. Seeded with the two matched pairs the roadmap names; more
#: are declarable with `register_cross_species_anchor`.
CROSS_SPECIES_ANCHORS = {}

#: Maps a drug name to the anchor key about that drug, so the partition
#: layer's gate can name the anchor in its refusal.
ANCHOR_KEY_BY_DRUG = {}


def register_cross_species_anchor(anchor, drug=None):
    """Declare an anchor. `drug` optionally keys it for partition-layer
    lookups (`ANCHOR_KEY_BY_DRUG`)."""
    if anchor.key in CROSS_SPECIES_ANCHORS:
        raise ValueError("anchor %r is already registered" % anchor.key)
    CROSS_SPECIES_ANCHORS[anchor.key] = anchor
    if drug is not None:
        ANCHOR_KEY_BY_DRUG[str(drug).lower()] = anchor.key
    return anchor


def declared_anchor(species, key=None):
    """The registered anchor covering `normalized species`, or None.

    `key` narrows the search to one anchor; without it every registered
    anchor is consulted and the first covering match wins (registration
    order). Covering is by SPECIES, not by drug: an anchor declares that
    this species pair's transfer error is quantified for THIS parameter,
    and `check_brain_partition` narrows by drug through
    `ANCHOR_KEY_BY_DRUG` before asking.
    """
    if species is None:
        return None
    normalized = normalize_species(species)
    if normalized == HUMAN:
        return None
    if key is not None:
        anchor = CROSS_SPECIES_ANCHORS.get(key)
        return anchor if anchor is not None and anchor.covers(normalized) \
            else None
    for anchor in CROSS_SPECIES_ANCHORS.values():
        if anchor.covers(normalized):
            return anchor
    return None


# --- the seeds --------------------------------------------------------------
#
# Numbers verified against the primary sources on 2026-08-23 (PubMed abstracts
# fetched verbatim; see this module's docstring). The morphine rat value's
# attribution differs from the roadmap's, which credits both numbers to
# Ederoth 2004; each number is cited here to the paper that measured it.

register_cross_species_anchor(CrossSpeciesAnchor(
    key="tpa023/occupancy-ec50",
    parameter=(
        "TPA023 plasma concentration at 50% benzodiazepine-binding-site "
        "occupancy (EC50)"
    ),
    measurements=[
        AnchorMeasurement(
            "human", "ng/mL",
            "Atack 2010, PMID 19779131 ([11C]flumazenil PET, humans)",
            value=9.0,
            note="slightly lower than the animal values, per the abstract",
        ),
        AnchorMeasurement(
            "rat", "ng/mL",
            "Atack 2010, PMID 19779131 (in vivo [3H]flumazenil binding / "
            "[11C]flumazenil microPET, rats)",
            interval=(19.0, 30.0),
            note="paper reports 19-30 ng/mL as one range spanning rat, "
                 "rhesus and baboon; the species are not separated",
        ),
        AnchorMeasurement(
            "rhesus", "ng/mL",
            "Atack 2010, PMID 19779131 ([123I]iomazenil scintigraphy, rhesus "
            "monkeys)",
            interval=(19.0, 30.0),
            note="same combined range as rat and baboon",
        ),
        AnchorMeasurement(
            "baboon", "ng/mL",
            "Atack 2010, PMID 19779131 ([11C]flumazenil PET, baboons)",
            interval=(19.0, 30.0),
            note="same combined range as rat and rhesus",
        ),
    ],
    note=(
        "The roadmap's named anchor for the benzodiazepine system: the human "
        "occupancy curve sits at roughly a third of the animal concentrations."
    ),
), drug="TPA023")

register_cross_species_anchor(CrossSpeciesAnchor(
    key="morphine/kp-uu",
    parameter="morphine Kp,uu (unbound brain : unbound plasma AUC ratio)",
    measurements=[
        AnchorMeasurement(
            "human", "dimensionless",
            "Ederoth 2004, PMID 15025740 (microdialysis in 'better' brain "
            "tissue of 7 traumatic-brain-injury patients)",
            value=0.64, interval=None,
            note="95% CI 0.40-0.87, reported as an interval and kept as one "
                 "in the source; the point value is the paper's own estimate",
        ),
        AnchorMeasurement(
            "rat", "dimensionless",
            "Tunblad 2003, PMID 12739770 (striatal microdialysis, 7 "
            "Sprague-Dawley rats, steady state)",
            value=0.29,
            note="+/- 0.07 sd; the roadmap credits this number to Ederoth "
                 "2004, which reports only the human value -- corrected here",
        ),
    ],
    note=(
        "Matrix caveat on the comparison: the rat ratio is unbound brain : "
        "unbound BLOOD, the human one unbound brain : unbound PLASMA. Both "
        "papers demonstrate active efflux (P-gp) at their respective BBBs, "
        "which is why the pair is the roadmap's template for a "
        "species-transfer factor on efflux-substrate Kp,uu (roadmap P14.3)."
    ),
), drug="morphine")


def anchors_report():
    """Every registered anchor, as text: the reporting surface the gate
    protects. This, and code like it, is what an anchor is FOR."""
    lines = [
        "Cross-species calibration anchors (P11.2)",
        "=" * 70,
        "",
        "Declared pairs measuring how far a rodent value is from the human",
        "one it must not stand in for. Reported and compared only:",
        "the species gate refuses every one of these as a prediction input.",
        "",
    ]
    if not CROSS_SPECIES_ANCHORS:
        lines.append("(none registered)")
    for key, anchor in CROSS_SPECIES_ANCHORS.items():
        lines.append(anchor.describe())
        ratios = anchor.transfer_ratios()
        if ratios:
            lines.append(
                "  transfer (non-human / human): %s"
                % ", ".join(
                    "%s %.4g" % (s, r) if isinstance(r, float)
                    else "%s %s" % (s, r)
                    for s, r in sorted(ratios.items())
                )
            )
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "UnsourcedParameterError",
    "HUMAN",
    "SPECIES_SYNONYMS",
    "normalize_species",
    "SourceMeasurement",
    "BindingRecord",
    "BINDING_RECORDS",
    "SITE_BINDING_RECORDS",
    "gate",
    "check_pk_model",
    "check_brain_partition",
    "check_binding_target",
    "check_receptor_site",
    "check_endpoint",
    "AnchorMeasurement",
    "CrossSpeciesAnchor",
    "CROSS_SPECIES_ANCHORS",
    "ANCHOR_KEY_BY_DRUG",
    "register_cross_species_anchor",
    "declared_anchor",
    "anchors_report",
]
