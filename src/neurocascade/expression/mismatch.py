"""P4.2: where a transcript map puts the receptor in the wrong region.

The failure mode this module is about
-------------------------------------

In situ hybridization measures mRNA, and mRNA is in the cell body. Receptor
protein is wherever the neuron trafficked it, and for a large class of
receptors that is the axon terminal, in a region that may be several
millimetres away and that transcribes none of the gene at all. For those
receptors the proportionality assumption of `density.py` does not merely
lose precision: it puts the receptor in the wrong node of the circuit, and a
circuit model is a machine for propagating that error into a behavioural
prediction.

This is not a hypothetical. Three of the receptor systems this codebase
already simulates or is scheduled to simulate are affected, and one of them
(the dopamine transporter) is affected totally: the regions with the most
transcript have the least protein-per-transcript and vice versa.

What this module does about it
------------------------------

Three things, and deliberately not a fourth.

1. **Records the documented cases**, gene by gene, with the source nucleus,
   the terminal field, a severity, and the primary literature that
   established the dissociation. `MISMATCHES` is that record.
2. **Quantifies them where both sides are measured in mouse.** For Slc6a3
   and Drd1 the transcript ratio between two regions and the *measured
   protein* ratio between the same two regions are both available, so the
   error the proportionality assumption makes is a number, not an adjective.
   `measured_displacement()` computes it from the data rather than storing
   it, so it cannot go stale against a change in the expression policy.
3. **Makes the affected genes refuse to convert silently.** A gene with a
   `severity == "terminal-field"` record raises out of
   `density.relative_density()` unless the caller passes
   `acknowledge_mismatch=True`. Refusing to be a silent default is the same
   move P4.1 made for `plane` and `combine`, applied to a hazard rather than
   a choice.

The fourth thing, deliberately not done: **no correction is applied.** The
obvious one - push each source nucleus's transcript out along the P3.1
connectome's projection weights - requires knowing what fraction of a
receptor pool is trafficked to terminals, per gene, and that fraction is not
published for any gene here. Multiplying by a plausible-looking fraction
would be inventing the number this project is not allowed to invent.
Correcting the maps is what Phase 6's projection-system abstraction is for,
and it will need transmitter-release and projection-density data that Phase
6 explicitly goes and gets. Until then the honest output is a flagged map
and a measured error bar on it.

Which genes are NOT here, and why that is a claim too
-----------------------------------------------------

The ionotropic receptors - `Grin*`, `Gria*`, `Gabra*`, `Gabrb*` - are
predominantly postsynaptic, on the dendrites of the cell that transcribes
them, which is the case the proportionality assumption is built for. They
are not in this registry, and their absence is an assertion about them and
not an oversight. (Presynaptic ionotropic receptors exist - kainate
autoreceptors, presynaptic NMDA - but they are a minority pool rather than
the dominant one, and no quantitative region-wise split is published.)
`Adora2a`, `Drd2` and `Htr1a` are the intermediate cases and are recorded
with severity `partial`.
"""

import numpy as np

#: How badly a transcript map misplaces the receptor.
#:
#:   "terminal-field"  the dominant protein pool is in a region that
#:                     transcribes little or none of the gene. A transcript
#:                     map is not a density map for this gene at all.
#:   "partial"         a substantial minority pool sits presynaptically in
#:                     regions the somata project to; the transcript map is
#:                     right about where most of the receptor is and wrong
#:                     about a specific, named set of regions.
SEVERITIES = ("terminal-field", "partial")


class TerminalFieldMismatch(object):
    """One gene's documented soma-versus-terminal dissociation."""

    __slots__ = (
        "symbol",
        "severity",
        "soma_regions",
        "terminal_regions",
        "mechanism",
        "citations",
        "measured",
    )

    def __init__(
        self,
        symbol,
        severity,
        soma_regions,
        terminal_regions,
        mechanism,
        citations,
        measured=None,
    ):
        if severity not in SEVERITIES:
            raise ValueError(
                "severity must be one of %s, got %r" % (list(SEVERITIES), severity)
            )
        self.symbol = symbol
        self.severity = severity
        self.soma_regions = tuple(soma_regions)
        self.terminal_regions = tuple(terminal_regions)
        self.mechanism = mechanism
        self.citations = tuple(citations)
        #: `(ligand_getter_name, soma acronym, terminal acronym)` when both
        #: sides are measured in mouse, else None. The ligand table lives in
        #: `autoradiography.py` and is looked up lazily so this module has no
        #: import-time dependency on it.
        self.measured = measured

    def __repr__(self):
        return "TerminalFieldMismatch(%r, %r)" % (self.symbol, self.severity)

    def summary(self):
        return (
            "%s: %s. Transcript sits in %s; a substantial receptor pool "
            "functions in %s. %s"
            % (
                self.symbol,
                self.severity,
                ", ".join(self.soma_regions),
                ", ".join(self.terminal_regions),
                self.mechanism,
            )
        )


MISMATCHES = (
    TerminalFieldMismatch(
        symbol="Slc6a3",
        severity="terminal-field",
        soma_regions=("SNc", "VTA"),
        terminal_regions=("CP", "ACB", "OT"),
        mechanism=(
            "The dopamine transporter is transcribed only by midbrain "
            "dopamine neurons and its job is reuptake at the release site, "
            "so the protein is overwhelmingly in the striatal neuropil "
            "those neurons project to. The transcript map and the protein "
            "map are close to disjoint."
        ),
        citations=(
            "Ciliax BJ, Heilman C, Demchyshyn LL, Pristupa ZB, Ince E, "
            "Hersch SM, Niznik HB, Levey AI. The dopamine transporter: "
            "immunochemical characterization and localization in brain. "
            "J Neurosci. 1995;15(3):1714-1723. PMID 7534339. (Rat; DAT "
            "immunoreactivity dense in striatal neuropil, somatodendritic "
            "in substantia nigra.)",
            "Yoo et al. 2010 (see autoradiography.YOO2010) for the mouse "
            "quantitative binding this case is measured against.",
        ),
        measured=("yoo2010_dat", "SNr", "CP"),
    ),
    TerminalFieldMismatch(
        symbol="Drd1",
        severity="partial",
        soma_regions=("CP", "ACB", "OT"),
        terminal_regions=("SNr", "GPi"),
        mechanism=(
            "D1 receptors are transcribed by direct-pathway striatal medium "
            "spiny neurons, and a large pool is transported down their axons "
            "to the striatonigral and striatoentopeduncular terminals. "
            "Substantia nigra therefore has D1 binding comparable to the "
            "cortex-and-above tier while transcribing almost none of the "
            "gene. The nigrostriatal bundle's own D1 binding - 91 fmol/mg in "
            "Yoo et al. 2010, in a fibre tract with no cell bodies at all - "
            "is this mechanism caught in transit."
        ),
        citations=(
            "Altar CA, Hauser K. Topography of substantia nigra innervation "
            "by D1 receptor-containing striatal neurons. Brain Res. "
            "1987;410(1):1-11. PMID 2953407. (Rat; striatal lesions deplete "
            "nigral D1 binding, placing the nigral sites on striatonigral "
            "terminals.)",
            "Yoo et al. 2010 (see autoradiography.YOO2010) for the mouse "
            "quantitative binding this case is measured against.",
        ),
        measured=("yoo2010_d1", "CP", "SNr"),
    ),
    TerminalFieldMismatch(
        symbol="Cnr1",
        severity="terminal-field",
        soma_regions=("CP", "HIP", "Isocortex"),
        terminal_regions=("SNr", "GPe", "GPi"),
        mechanism=(
            "CB1 is transcribed by striatal projection neurons and by "
            "cortical and hippocampal interneurons, and functions almost "
            "entirely as a presynaptic terminal receptor. Pallidum and "
            "substantia nigra pars reticulata carry the densest CB1 binding "
            "in the brain and essentially no CB1 transcript."
        ),
        citations=(
            "Herkenham M, Lynn AB, de Costa BR, Richfield EK. Neuronal "
            "localization of cannabinoid receptors in the basal ganglia of "
            "the rat. Brain Res. 1991;547(2):267-274. PMID 1909204. (Rat; "
            "striatal lesion depletes pallidal and nigral CB1 binding.)",
            "Mailleux P, Vanderhaeghen JJ. Distribution of neuronal "
            "cannabinoid receptor in the adult rat brain: a comparative "
            "receptor binding radioautography and in situ hybridization "
            "histochemistry. Neuroscience. 1992;48(3):655-668. PMID 1376455. "
            "(Rat; the mRNA and binding maps side by side.)",
        ),
        measured=None,
    ),
    TerminalFieldMismatch(
        symbol="Slc6a4",
        severity="terminal-field",
        soma_regions=("DR", "RAmb"),
        terminal_regions=("Isocortex", "STR", "HIP", "TH"),
        mechanism=(
            "Same architecture as the dopamine transporter: SERT is "
            "transcribed only by raphe serotonin neurons and functions on "
            "their axons throughout the forebrain."
        ),
        citations=(
            "Sur C, Betz H, Schloss P. Immunocytochemical detection of the "
            "serotonin transporter in rat brain. Neuroscience. "
            "1996;73(1):217-231. PMID 8783244.",
        ),
        measured=None,
    ),
    TerminalFieldMismatch(
        symbol="Slc6a2",
        severity="terminal-field",
        soma_regions=("LC",),
        terminal_regions=("Isocortex", "HIP", "TH"),
        mechanism=(
            "The noradrenaline transporter is transcribed by locus "
            "coeruleus neurons and functions on their terminals brain-wide."
        ),
        citations=(
            "Schroeter S, Apparsundaram S, Wiley RG, Miner LH, Sesack SR, "
            "Blakely RD. Immunolocalization of the cocaine- and "
            "antidepressant-sensitive l-norepinephrine transporter. "
            "J Comp Neurol. 2000;420(2):211-232. PMID 10753308.",
        ),
        measured=None,
    ),
    TerminalFieldMismatch(
        symbol="Oprm1",
        severity="partial",
        soma_regions=("MH",),
        terminal_regions=("IPN",),
        mechanism=(
            "The medial habenula transcribes mu-opioid receptor at the "
            "highest level in the brain and projects, via the fasciculus "
            "retroflexus, to the interpeduncular nucleus, where a large "
            "part of that protein ends up. The rest of the Oprm1 map - "
            "midline thalamus, striatal patches, PAG, LC - is not affected "
            "this way, which is why the severity is partial and not "
            "terminal-field."
        ),
        citations=(
            "Gardon O, Faget L, Chu Sin Chung P, Matifas A, Massotte D, "
            "Kieffer BL. Expression of mu opioid receptor in dorsal "
            "diencephalic conduction system: new insights for the medial "
            "habenula. Neuroscience. 2014;277:595-609. PMID 25086313. "
            "(Mouse; MOR protein along the fasciculus retroflexus and in "
            "the interpeduncular nucleus.)",
        ),
        measured=None,
    ),
    TerminalFieldMismatch(
        symbol="Adra2a",
        severity="partial",
        soma_regions=("LC", "Isocortex"),
        terminal_regions=("Isocortex", "HIP"),
        mechanism=(
            "alpha-2A is both a postsynaptic cortical receptor, where the "
            "transcript map is right, and the noradrenergic autoreceptor on "
            "locus coeruleus terminals, where it is not. The two pools are "
            "not separable in a binding assay and no published region-wise "
            "split exists, so the size of this one is unquantified."
        ),
        citations=(
            "Schroeter et al. 2000, PMID 10753308, for the LC terminal "
            "field the autoreceptor pool follows.",
        ),
        measured=None,
    ),
    TerminalFieldMismatch(
        symbol="Drd2",
        severity="partial",
        soma_regions=("CP", "ACB", "SNc", "VTA"),
        terminal_regions=("CP", "ACB"),
        mechanism=(
            "Striatal D2 is mostly postsynaptic on indirect-pathway medium "
            "spiny neurons, which transcribe it, so the dominant pool is "
            "where the transcript is. The minority pool is the D2 "
            "autoreceptor on nigrostriatal terminals, transcribed in SNc/VTA "
            "and functioning in the striatum - but SNc and VTA transcribe "
            "Drd2 as well, so unlike the transporters both ends of the "
            "projection appear in the transcript map. This is the "
            "best-behaved of the affected genes and it is the receptor the "
            "haloperidol scenario uses."
        ),
        citations=(
            "Recorded as partial on the strength of the transcript map "
            "itself: Drd2 is expressed in SNc and VTA as well as in the "
            "striatum, so the autoreceptor pool's source is not missing "
            "from the map the way DAT's terminal field is.",
        ),
        measured=None,
    ),
)

MISMATCHES_BY_SYMBOL = {record.symbol: record for record in MISMATCHES}

#: The genes whose transcript map is not a density map at all.
TERMINAL_FIELD_SYMBOLS = tuple(
    record.symbol for record in MISMATCHES if record.severity == "terminal-field"
)


def mismatch_for(symbol):
    """The record for a gene, or None if it has no documented mismatch."""
    return MISMATCHES_BY_SYMBOL.get(symbol)


class Displacement(object):
    """How far the proportionality assumption misplaces one receptor pool.

    `factor` is the ratio of ratios: how much bigger the transcript's
    soma-to-terminal ratio is than the measured protein's. A factor of 1
    would mean the transcript map already has the two regions in the right
    proportion; the cases here are two orders of magnitude off.
    """

    def __init__(
        self,
        symbol,
        soma,
        terminal,
        soma_energy,
        terminal_energy,
        soma_density,
        terminal_density,
        ligand,
        dataset,
    ):
        self.symbol = symbol
        self.soma = soma
        self.terminal = terminal
        self.soma_energy = float(soma_energy)
        self.terminal_energy = float(terminal_energy)
        self.soma_density = float(soma_density)
        self.terminal_density = float(terminal_density)
        self.ligand = ligand
        self.dataset = dataset

    @property
    def energy_ratio(self):
        """Transcript, soma / terminal."""
        return self.soma_energy / self.terminal_energy

    @property
    def density_ratio(self):
        """Measured protein, soma / terminal."""
        return self.soma_density / self.terminal_density

    @property
    def factor(self):
        return self.energy_ratio / self.density_ratio

    def __repr__(self):
        return "Displacement(%s, %s/%s, factor=%.3g)" % (
            self.symbol,
            self.soma,
            self.terminal,
            self.factor,
        )

    def describe(self):
        return (
            "%s: transcript puts %.3g times as much in %s as in %s; "
            "%s binding puts %.3g times as much. Assuming density is "
            "proportional to transcript misstates the %s:%s ratio by a "
            "factor of %.3g."
            % (
                self.symbol,
                self.energy_ratio,
                self.soma,
                self.terminal,
                self.ligand,
                self.density_ratio,
                self.soma,
                self.terminal,
                self.factor,
            )
        )


def measured_displacement(expression, symbol, ontology=None):
    """Quantify one gene's mismatch against measured mouse binding.

    `expression` is a `RegionalExpression` that covers both regions of the
    gene's `measured` pair. Returns None for a gene with no measured pair -
    which is most of them, and is why the registry records severity in words
    as well.
    """
    from . import autoradiography

    record = mismatch_for(symbol)
    if record is None or record.measured is None:
        return None
    getter_name, soma, terminal = record.measured
    binding = getattr(autoradiography, getter_name)()
    densities = binding.mapped()
    if soma not in densities:
        # The one case the region mapping deliberately leaves ambiguous.
        if soma in autoradiography.SUBSTANTIA_NIGRA_ALTERNATIVES:
            densities[soma] = binding.values["Substantia nigra"]
        else:
            raise KeyError(
                "%s has no published density for %s" % (getter_name, soma)
            )
    if terminal not in densities:
        if terminal in autoradiography.SUBSTANTIA_NIGRA_ALTERNATIVES:
            densities[terminal] = binding.values["Substantia nigra"]
        else:
            raise KeyError(
                "%s has no published density for %s" % (getter_name, terminal)
            )

    if ontology is None:
        ontology = expression.ontology
    if ontology is None:
        raise ValueError(
            "an ontology is needed to turn %s and %s into structure ids"
            % (soma, terminal)
        )
    energies = {}
    row = expression.for_gene(symbol)
    ids = list(expression.structure_ids)
    for acronym in (soma, terminal):
        structure_id = ontology.id_for_acronym(acronym)
        if structure_id not in ids:
            raise KeyError(
                "%s (%d) is not in this expression matrix, so the %s "
                "displacement cannot be measured on it"
                % (acronym, structure_id, symbol)
            )
        value = row[ids.index(structure_id)]
        if not np.isfinite(value):
            raise ValueError(
                "%s has no expression value in %s under this policy"
                % (symbol, acronym)
            )
        energies[acronym] = float(value)

    return Displacement(
        symbol=symbol,
        soma=soma,
        terminal=terminal,
        soma_energy=energies[soma],
        terminal_energy=energies[terminal],
        soma_density=densities[soma],
        terminal_density=densities[terminal],
        ligand=binding.ligand,
        dataset=binding.dataset,
    )
