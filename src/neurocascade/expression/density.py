"""P4.2: expression energy -> the density scale the receptor layer consumes.

The assumption, stated as an assumption
---------------------------------------

    ASSUMPTION (P4.2, unproven, and false in named cases):
    within one gene, receptor density in a region is proportional to that
    gene's Allen ISH expression energy in that region.

        density(region) = k_gene * energy(region)

    with a single constant k_gene per gene, unknown in absolute units unless
    a measured density is available to fit it against.

Everything this module produces rests on that line. It is an assumption and
not a finding, for reasons that are individually well established:

- **mRNA is not protein.** Transcript abundance and protein abundance are
  coupled through translation rate, protein half-life and trafficking, none
  of which is uniform across regions. Where the two have been measured
  together in brain tissue, they correlate but not tightly.
- **Protein is not binding sites.** A receptor in an intracellular vesicle,
  or an unassembled subunit, is protein and is not a target a drug can
  occupy. GABA-A is a pentamer: five subunit transcripts and one binding
  site whose pharmacology depends on which five.
- **The receptor may not be in the region that transcribes it.** This is the
  sharp one and it has its own module, `mismatch.py`.
- **Expression energy is not comparable between genes.** Probe, riboprobe
  length and segmentation threshold differ per experiment, so `k_gene` is
  per gene and there is no meaning to comparing two genes' energies. Every
  function here works within one gene, and none of them offers a
  cross-gene density.

So the constant is `k_gene`, the relation is assumed linear through the
origin, and `validation.py` measures how badly both halves of that are
wrong for the one receptor where an independent mouse density map exists.
The short version of that measurement, for the reader who stops here: over
ten regions with published mouse D1 densities the *ranks* agree well
(Spearman ~0.7) and the *proportionality* does not (a fit through the
origin explains about a third of the variance, and the log-log slope is
about 0.5, not 1). The map is much better as an ordering than as a scale.

What the receptor layer actually consumes
-----------------------------------------

`ReceptorSite` in `receptor.py` holds *fractional* occupancy: the state is
the share of a region's receptor pool bound by each ligand, and it is
dimensionless. Nothing in the binding ODE needs a density at all, and that
is not an oversight - competitive occupancy really is density-independent
when ligand concentration is not depleted by binding.

Density enters one step later. `Simulation.rhs` adds `gain * activation` to
a node's drive, and that single number folds together two things: how much
receptor is in the region, and how much a unit of receptor activation
changes that node's net input given the circuit around it. P4.2 supplies
the first factor and refuses to supply the second:

    coupling[node] = mechanism_gain[node] * rho[region]

`rho` is what this module returns: a dimensionless, per-region relative
density, normalized by an explicitly chosen policy. `mechanism_gain` stays
whatever it already is - illustrative in `scenarios.py`, fitted in P2.2 -
and stays labelled that way. `ReceptorDensityMap.as_coupling()` does the
multiplication so the factorization is written down somewhere executable
rather than only in a docstring.

The reason for that split is a lesson already in this codebase. The
multi-region mu-opioid scenario gives LC the *smallest* coupling gain of
its three targets even though LC has the *highest* published mu-receptor
density of the three, because LC's receptor acts directly on the node being
read out while the other two act through an interposed
interneuron-disinhibition step that amplifies them. Density and gain are not
the same axis. A P4.2 that returned "the coupling constant" would have
silently asserted they were.

Absolute units, where and only where they are earned
----------------------------------------------------

`relative_density()` returns a dimensionless ratio and says so. A map in
fmol/mg requires `k_gene`, and `k_gene` requires an independent measured
density for that specific gene in that specific species. Exactly one gene
in this project has one (`Drd1`, against Yoo et al. 2010), so exactly one
gene can have an absolute scale, it is fitted rather than assumed, and
`AbsoluteScale` refuses to be applied to any other gene. Extending a
constant fitted on D1 to Oprm1 would be inventing a number.
"""

import numpy as np

from . import mismatch as _mismatch

#: How a gene's row is turned into a dimensionless relative density.
#:
#:   "max"               divide by the largest measured region: rho in
#:                       [0, 1], 1 at the peak region
#:   "mean"              divide by the mean over measured regions: rho = 1
#:                       is "an average region for this receptor"
#:   "reference-region"  divide by one named region, which is then 1.0 by
#:                       construction. Use when a scenario has a region
#:                       whose coupling gain is already calibrated and the
#:                       others should be expressed relative to it.
#:   "none"              pass the energy through unchanged. Not a density,
#:                       and the returned map says so in its units.
NORMALIZATIONS = ("max", "mean", "reference-region", "none")

#: The exponent of the assumed density(energy) relation. 1.0 IS the
#: proportionality assumption; it is a named constant so that the one place
#: the assumption enters the arithmetic is greppable.
PROPORTIONALITY_EXPONENT = 1.0

ASSUMPTION = (
    "Within one gene, receptor density in a region is assumed proportional "
    "to Allen ISH expression energy in that region: density = k_gene * "
    "energy, k_gene unknown and per gene. mRNA is not protein, protein is "
    "not binding sites, and for the genes in mismatch.MISMATCHES the "
    "protein is not even in the same region as the transcript. This is an "
    "assumption, not a measurement; validation.py measures its error where "
    "an independent mouse density map exists."
)


class AbsoluteScale(object):
    """A fitted `k_gene`, and the published data it was fitted against.

    Constructed only by `validation.fit_absolute_scale()`, from a comparison
    against measured densities. There is no constructor path that lets a
    caller assert a scale, because a scale nobody measured is exactly the
    invented number this project forbids.
    """

    def __init__(self, symbol, factor, units, n_regions, r_squared,
                 dataset, ligand, note=""):
        if factor <= 0.0 or not np.isfinite(factor):
            raise ValueError("k_gene must be finite and positive, got %r" % (factor,))
        self.symbol = symbol
        self.factor = float(factor)
        self.units = units
        self.n_regions = int(n_regions)
        self.r_squared = float(r_squared)
        self.dataset = dataset
        self.ligand = ligand
        self.note = note

    def __repr__(self):
        return "AbsoluteScale(%r, %.4g %s per energy unit, R2=%.3f)" % (
            self.symbol,
            self.factor,
            self.units,
            self.r_squared,
        )


class ReceptorDensityMap(object):
    """One gene's per-region relative density, and how it was made.

    `values[i]` is the relative density of `symbol` in `structure_ids[i]`,
    NaN where the ISH data does not measure that region. NaN is not zero
    here for the same reason it is not zero in `RegionalExpression`: zero is
    the claim that a drug acting at this receptor has no target in the
    region, and "not measured" is not that claim.
    """

    def __init__(
        self,
        values,
        structure_ids,
        symbol,
        normalization,
        units,
        source,
        exponent=PROPORTIONALITY_EXPONENT,
        reference=None,
        mismatch=None,
        ontology=None,
        absolute_scale=None,
    ):
        self.values = np.asarray(values, dtype=float)
        self.structure_ids = np.asarray(structure_ids, dtype=int)
        self.symbol = symbol
        self.normalization = normalization
        self.units = units
        #: The `RegionalExpression` this came from, so plane, combine rule
        #: and contributing dataset ids stay attached to the density.
        self.source = source
        self.exponent = float(exponent)
        self.reference = reference
        self.mismatch = mismatch
        self.ontology = ontology
        self.absolute_scale = absolute_scale
        if self.values.shape != self.structure_ids.shape:
            raise ValueError(
                "%d values for %d regions"
                % (self.values.size, self.structure_ids.size)
            )
        finite = np.isfinite(self.values)
        if np.any(self.values[finite] < 0.0):
            raise ValueError("negative relative density; this is a bug")

    def __repr__(self):
        return "ReceptorDensityMap(%r, %d regions, normalization=%r%s)" % (
            self.symbol,
            self.structure_ids.size,
            self.normalization,
            ", FLAGGED" if self.mismatch is not None else "",
        )

    @property
    def assumption(self):
        return ASSUMPTION

    def index_of(self, structure_id):
        matches = np.nonzero(self.structure_ids == int(structure_id))[0]
        if matches.size == 0:
            raise KeyError(
                "structure %d is not in this map" % int(structure_id)
            )
        return int(matches[0])

    def for_region(self, structure_id):
        return float(self.values[self.index_of(structure_id)])

    def measured(self):
        """`(structure_ids, values)` for the regions that have a value."""
        finite = np.isfinite(self.values)
        return self.structure_ids[finite], self.values[finite]

    def top(self, n=10):
        """The n highest regions, as `[(structure_id, value)]`."""
        ids, values = self.measured()
        order = np.argsort(values)[::-1][:n]
        return [(int(ids[i]), float(values[i])) for i in order]

    def acronyms(self):
        if self.ontology is None:
            raise ValueError("this map was built without an ontology")
        return [self.ontology.acronym_for_id(int(i)) for i in self.structure_ids]

    def as_coupling(self, mechanism_gain, nodes):
        """`{node: mechanism_gain * rho[region]}`, the receptor layer's input.

        `nodes` maps a circuit node name to the CCF structure id it stands
        for. `mechanism_gain` is a single number or a per-node dict: it is
        the OTHER factor, how much a unit of receptor activation moves that
        node given the circuit around it, and it is not P4.2's to supply.
        See this module's docstring for why the two factors are separate.

        A region with no measurement raises. Emitting 0.0 for it would turn
        "the ISH data does not reach this region" into "a drug acting here
        has no target", which is a scientific claim and a false one.
        """
        gains = mechanism_gain
        if not isinstance(gains, dict):
            gains = {node: mechanism_gain for node in nodes}
        coupling = {}
        for node, structure_id in nodes.items():
            if node not in gains:
                raise KeyError("no mechanism gain for node %r" % (node,))
            value = self.for_region(structure_id)
            if not np.isfinite(value):
                raise ValueError(
                    "%s has no measured expression in structure %d (node "
                    "%r), so this map cannot supply its density factor. "
                    "Substituting zero would assert the receptor is absent "
                    "there; it asserts nothing."
                    % (self.symbol, int(structure_id), node)
                )
            coupling[node] = gains[node] * value
        return coupling

    def to_absolute(self, scale):
        """Multiply through by a fitted `k_gene`. Same gene only."""
        if scale.symbol != self.symbol:
            raise ValueError(
                "the absolute scale was fitted for %s and this map is %s. "
                "Expression energy is not comparable between genes (probe "
                "and segmentation threshold differ per experiment), so a "
                "k_gene fitted on one gene says nothing about another."
                % (scale.symbol, self.symbol)
            )
        if self.normalization != "none":
            raise ValueError(
                "an absolute scale converts expression energy, not a "
                "normalized ratio; build the map with normalization='none' "
                "first (this one is %r)" % (self.normalization,)
            )
        return ReceptorDensityMap(
            values=self.values * scale.factor,
            structure_ids=self.structure_ids,
            symbol=self.symbol,
            normalization=self.normalization,
            units=scale.units,
            source=self.source,
            exponent=self.exponent,
            reference=self.reference,
            mismatch=self.mismatch,
            ontology=self.ontology,
            absolute_scale=scale,
        )


def relative_density(
    expression,
    symbol,
    normalization=None,
    reference=None,
    exponent=PROPORTIONALITY_EXPONENT,
    acknowledge_mismatch=False,
):
    """Convert one gene's expression row into a relative density map.

    `normalization` is required; see `NORMALIZATIONS`. There is no default
    for the same reason `energy_matrix()` has no default plane: the choice
    changes what the number means (peak-relative, average-relative, or
    relative to one named region) and a loader that picked one would hide
    the choice.

    `acknowledge_mismatch` must be True for a gene whose receptor protein
    is documented to sit in a different region from its transcript
    (`mismatch.TERMINAL_FIELD_SYMBOLS`). The flag does not change the
    arithmetic - there is no correction to apply, see `mismatch.py` - it
    exists so that using a dopamine-transporter transcript map as a
    dopamine-transporter density map is something a caller had to type.
    """
    if normalization is None:
        raise TypeError(
            "relative_density() requires normalization=. There is no "
            "default because 'relative to the peak region', 'relative to "
            "the average region' and 'relative to one named region' are "
            "three different quantities and the difference is invisible in "
            "the output. normalization in %s." % (list(NORMALIZATIONS),)
        )
    if normalization not in NORMALIZATIONS:
        raise ValueError(
            "normalization must be one of %s, got %r"
            % (list(NORMALIZATIONS), normalization)
        )
    if not np.isfinite(exponent) or exponent <= 0.0:
        raise ValueError("exponent must be finite and positive, got %r" % (exponent,))

    record = _mismatch.mismatch_for(symbol)
    if (
        record is not None
        and record.severity == "terminal-field"
        and not acknowledge_mismatch
    ):
        raise ValueError(
            "%s's receptor protein is not where its transcript is. %s "
            "This map is still the best available and there is no published "
            "correction to apply, so it is not refused outright - but it is "
            "not a density map for this gene, and callers have to say they "
            "know: pass acknowledge_mismatch=True. Citations in "
            "mismatch.MISMATCHES_BY_SYMBOL[%r]."
            % (symbol, record.summary(), symbol)
        )

    row = np.array(expression.for_gene(symbol), dtype=float)
    finite = np.isfinite(row)
    if not np.any(finite):
        raise ValueError(
            "%s has no measured region in this matrix (plane=%r, "
            "combine=%r), so there is nothing to normalize"
            % (symbol, expression.plane, expression.combine)
        )

    values = np.full(row.shape, np.nan)
    if exponent == 1.0:
        values[finite] = row[finite]
    else:
        values[finite] = row[finite] ** exponent

    reference_id = None
    if normalization == "max":
        divisor = np.nanmax(values)
        units = "dimensionless, relative to the peak measured region"
    elif normalization == "mean":
        divisor = np.nanmean(values)
        units = "dimensionless, relative to the mean measured region"
    elif normalization == "reference-region":
        if reference is None:
            raise TypeError(
                "normalization='reference-region' needs reference=<structure id>"
            )
        reference_id = int(reference)
        ids = [int(i) for i in expression.structure_ids]
        if reference_id not in ids:
            raise KeyError(
                "reference region %d is not in this matrix" % reference_id
            )
        divisor = values[ids.index(reference_id)]
        if not np.isfinite(divisor) or divisor <= 0.0:
            raise ValueError(
                "reference region %d has no usable expression value (%r), "
                "so everything divided by it would be meaningless"
                % (reference_id, divisor)
            )
        units = "dimensionless, relative to structure %d" % reference_id
    else:
        divisor = 1.0
        units = expression.units

    if divisor <= 0.0:
        raise ValueError(
            "the normalizing divisor is %r; %s has no positive expression "
            "anywhere in this matrix" % (divisor, symbol)
        )

    return ReceptorDensityMap(
        values=values / divisor,
        structure_ids=expression.structure_ids,
        symbol=symbol,
        normalization=normalization,
        units=units,
        source=expression,
        exponent=exponent,
        reference=reference_id,
        mismatch=record,
        ontology=expression.ontology,
    )
