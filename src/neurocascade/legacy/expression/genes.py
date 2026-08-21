"""P4.1: the receptor genes this project needs an expression map for, and
what each one encodes.

Why a gene table and not a bare list of symbols
-----------------------------------------------

The roadmap names receptors ("mu-opioid", "GABA-A", "D2"); the Allen atlas
is indexed by genes. The step between the two is not mechanical and it is
where a whole-brain receptor map can go quietly wrong:

* **A receptor is not always a gene.** GABA-A is a pentamer assembled from
  a family of subunit genes; there is no `Gabaa` to look up. `Gabra1` and
  `Gabra5` expression are different maps of different subunit pools, and
  the benzodiazepine site this codebase already models exists only on
  alpha1/2/3/5-containing pentamers (alpha4- and alpha6-containing ones are
  diazepam-insensitive). So the alpha subunits are ingested separately and
  which of them a benzodiazepine parameter may use is a P4.2/P5 question,
  not something to average away here.
* **A gene is not always the receptor's name.** The dopamine transporter is
  `Slc6a3`, the serotonin transporter `Slc6a4`, the noradrenaline
  transporter `Slc6a2`. Nothing in the symbol says so.
* **Symbols move.** The mouse D1 receptor gene was `Drd1a` for years and is
  now `Drd1`; the Allen gene table carries the current symbol only, and a
  query for `Drd1a` returns zero datasets rather than an error. This table
  therefore records the NCBI Entrez gene id alongside the symbol, and the
  fetch script checks that the symbol the Allen API resolved carries the
  expected Entrez id. A silent symbol drift would otherwise produce an
  empty map that looks like a real "gene not expressed" result.

Each entry also records which roadmap phase actually consumes the map, so
the difference between "the model uses this today" and "Phase 5 will" stays
visible rather than being flattened into one undifferentiated matrix.

What this table does NOT claim
------------------------------

mRNA is not protein and neither is receptor density. Nothing in this
module converts expression energy into a density, a Bmax, or a
concentration; P4.2 owns that conversion and owns stating the
proportionality assumption as an assumption. This table is the list of
things to fetch and the reason each is on it.
"""


class ReceptorGene(object):
    """One gene, what it encodes, and why this project wants its map."""

    __slots__ = ("symbol", "entrez_id", "protein", "system", "phase", "note")

    def __init__(self, symbol, entrez_id, protein, system, phase, note=""):
        self.symbol = symbol
        self.entrez_id = entrez_id
        self.protein = protein
        self.system = system
        self.phase = phase
        self.note = note

    def __repr__(self):
        return "ReceptorGene(%r, %r)" % (self.symbol, self.protein)


#: Mouse (Mus musculus) receptor and transporter genes, with NCBI Entrez ids.
#: `phase` is the roadmap phase that consumes the map: "in model" means a
#: receptor this codebase already simulates.
RECEPTOR_GENES = (
    # --- opioid ------------------------------------------------------------
    ReceptorGene(
        "Oprm1", 18390, "mu-opioid receptor (MOR)", "opioid", "in model",
        "the receptor of the fentanyl scenarios; roadmap P4.3 expects a "
        "thalamus/striatum-dominant map",
    ),
    ReceptorGene(
        "Oprk1", 18387, "kappa-opioid receptor (KOR)", "opioid", "P5.7"
    ),
    ReceptorGene(
        "Oprd1", 18386, "delta-opioid receptor (DOR)", "opioid", "P5.7"
    ),
    # --- dopamine ----------------------------------------------------------
    ReceptorGene(
        "Drd1", 13488, "dopamine D1 receptor", "dopamine", "P5.2",
        "Allen's current symbol; the older mouse symbol Drd1a resolves to "
        "nothing in the Allen gene table",
    ),
    ReceptorGene(
        "Drd2", 13489, "dopamine D2 receptor", "dopamine", "in model",
        "the receptor of the haloperidol scenario; P4.3 expects a "
        "striatum-dominant map",
    ),
    # --- GABA-A ------------------------------------------------------------
    ReceptorGene(
        "Gabra1", 14394, "GABA-A receptor alpha1 subunit", "gaba_a", "in model",
        "benzodiazepine-sensitive; P4.3 expects cortex/cerebellum dominance",
    ),
    ReceptorGene(
        "Gabra2", 14395, "GABA-A receptor alpha2 subunit", "gaba_a", "in model",
        "benzodiazepine-sensitive",
    ),
    ReceptorGene(
        "Gabra3", 14396, "GABA-A receptor alpha3 subunit", "gaba_a", "in model",
        "benzodiazepine-sensitive",
    ),
    ReceptorGene(
        "Gabra4", 14397, "GABA-A receptor alpha4 subunit", "gaba_a", "in model",
        "classical-benzodiazepine-INsensitive; ingested so that P4.2 can "
        "exclude it from a diazepam-site map rather than silently include it",
    ),
    ReceptorGene(
        "Gabra5", 110886, "GABA-A receptor alpha5 subunit", "gaba_a", "in model",
        "benzodiazepine-sensitive",
    ),
    ReceptorGene(
        "Gabrb1", 14400, "GABA-A receptor beta1 subunit", "gaba_a", "in model"
    ),
    ReceptorGene(
        "Gabrb2", 14401, "GABA-A receptor beta2 subunit", "gaba_a", "in model"
    ),
    ReceptorGene(
        "Gabrb3", 14402, "GABA-A receptor beta3 subunit", "gaba_a", "in model"
    ),
    # --- serotonin ---------------------------------------------------------
    ReceptorGene("Htr1a", 15550, "5-HT1A receptor", "serotonin", "P5.1"),
    ReceptorGene("Htr2a", 15558, "5-HT2A receptor", "serotonin", "P5.1"),
    # --- acetylcholine -----------------------------------------------------
    ReceptorGene("Chrm1", 12669, "muscarinic M1 receptor", "acetylcholine", "P5.4"),
    ReceptorGene("Chrm2", 243764, "muscarinic M2 receptor", "acetylcholine", "P5.4"),
    ReceptorGene("Chrm3", 12671, "muscarinic M3 receptor", "acetylcholine", "P5.4"),
    ReceptorGene("Chrm4", 12672, "muscarinic M4 receptor", "acetylcholine", "P5.4"),
    ReceptorGene("Chrm5", 213788, "muscarinic M5 receptor", "acetylcholine", "P5.4"),
    ReceptorGene(
        "Chrna4", 11438, "nicotinic acetylcholine receptor alpha4 subunit",
        "acetylcholine", "P5.4",
        "the alpha4 of the alpha4-beta2 heteromer, the high-affinity "
        "nicotine site",
    ),
    ReceptorGene(
        "Chrna7", 11441, "nicotinic acetylcholine receptor alpha7 subunit",
        "acetylcholine", "P5.4", "the homomeric alpha7 receptor",
    ),
    # --- noradrenaline -----------------------------------------------------
    ReceptorGene("Adra1a", 11549, "adrenergic alpha-1A receptor", "noradrenaline", "P5.3"),
    ReceptorGene(
        "Adra1b", 11548, "adrenergic alpha-1B receptor", "noradrenaline", "P5.3",
        "no ISH experiment exists in the Allen Mouse Brain Atlas; see "
        "sources.GENES_WITHOUT_DATA",
    ),
    ReceptorGene("Adra1d", 11550, "adrenergic alpha-1D receptor", "noradrenaline", "P5.3"),
    ReceptorGene("Adra2a", 11551, "adrenergic alpha-2A receptor", "noradrenaline", "P5.3"),
    ReceptorGene("Adra2b", 11552, "adrenergic alpha-2B receptor", "noradrenaline", "P5.3"),
    ReceptorGene("Adra2c", 11553, "adrenergic alpha-2C receptor", "noradrenaline", "P5.3"),
    ReceptorGene("Adrb1", 11554, "adrenergic beta-1 receptor", "noradrenaline", "P5.3"),
    ReceptorGene("Adrb2", 11555, "adrenergic beta-2 receptor", "noradrenaline", "P5.3"),
    # --- glutamate ---------------------------------------------------------
    ReceptorGene(
        "Grin1", 14810, "NMDA receptor GluN1 subunit", "glutamate", "P5.5",
        "obligatory subunit of every NMDA receptor",
    ),
    ReceptorGene("Grin2a", 14811, "NMDA receptor GluN2A subunit", "glutamate", "P5.5"),
    ReceptorGene("Grin2b", 14812, "NMDA receptor GluN2B subunit", "glutamate", "P5.5"),
    ReceptorGene("Gria1", 14799, "AMPA receptor GluA1 subunit", "glutamate", "P5.5"),
    ReceptorGene("Gria2", 14800, "AMPA receptor GluA2 subunit", "glutamate", "P5.5"),
    ReceptorGene("Gria3", 53623, "AMPA receptor GluA3 subunit", "glutamate", "P5.5"),
    ReceptorGene("Gria4", 14802, "AMPA receptor GluA4 subunit", "glutamate", "P5.5"),
    # --- histamine, cannabinoid, adenosine ---------------------------------
    ReceptorGene("Hrh1", 15465, "histamine H1 receptor", "histamine", "P5.6"),
    ReceptorGene("Cnr1", 12801, "cannabinoid CB1 receptor", "cannabinoid", "P5.6"),
    ReceptorGene("Adora2a", 11540, "adenosine A2A receptor", "adenosine", "P5.6"),
    # --- monoamine transporters -------------------------------------------
    ReceptorGene(
        "Slc6a2", 20538, "noradrenaline transporter (NET)", "transporter", "P5.8"
    ),
    ReceptorGene(
        "Slc6a3", 13162, "dopamine transporter (DAT)", "transporter", "P5.8",
        "P4.3-style expectation: confined to the dopaminergic somata (VTA, "
        "SNc), not to their targets",
    ),
    ReceptorGene(
        "Slc6a4", 15567, "serotonin transporter (SERT)", "transporter", "P5.8"
    ),
)

GENES_BY_SYMBOL = {gene.symbol: gene for gene in RECEPTOR_GENES}

#: Distinct receptor systems represented, in table order.
SYSTEMS = tuple(
    dict.fromkeys(gene.system for gene in RECEPTOR_GENES)
)


def gene(symbol):
    """Look up one gene, with an error that lists what is available."""
    try:
        return GENES_BY_SYMBOL[symbol]
    except KeyError:
        raise KeyError(
            "%r is not a gene this project ingests; the table has: %s"
            % (symbol, ", ".join(sorted(GENES_BY_SYMBOL)))
        )


def genes_for_system(system):
    """Every gene of one receptor system, in table order."""
    return [g for g in RECEPTOR_GENES if g.system == system]


def symbols():
    """Every gene symbol, in table order."""
    return [g.symbol for g in RECEPTOR_GENES]
