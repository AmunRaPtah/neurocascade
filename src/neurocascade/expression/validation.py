"""P4.2: measuring how wrong the proportionality assumption is.

`density.py` assumes density = k_gene * energy. This module compares that
assumption against published mouse receptor autoradiography, region by
region, for one gene at a time, and reports the error rather than a verdict.

Three numbers, because they answer three different questions
------------------------------------------------------------

A single correlation coefficient would hide the thing that actually matters
here, which is that the map is much better at *ordering* regions than at
*scaling* them:

- **Spearman rho** - does the map rank the regions correctly? This is what a
  circuit model needs first: which nodes deserve a large coupling and which
  a small one.
- **R^2 of a fit through the origin** - is the relation proportional, with
  one constant? This is what an absolute density map needs, and it is the
  one that fails.
- **The log-log slope** - if the relation is a power law, what is the
  exponent? A slope of 1 is proportionality. A slope below 1 says the
  protein range is compressed relative to the transcript range, which is
  what a saturating translation-and-trafficking chain would do, and is also
  what an autoradiographic detection floor in the low-density regions would
  fake. Both explanations are live and this data cannot separate them, so
  the exponent is reported and not applied.

Nothing here rescales, drops an outlier, or widens a tolerance to make a
comparison look better. `worst()` exists specifically so the largest
disagreements are the first thing a reader sees.
"""

import numpy as np
from scipy import stats

from . import autoradiography as _auto
from .density import AbsoluteScale


class DensityComparison(object):
    """One gene's transcript map against one published binding table."""

    def __init__(self, symbol, binding, acronyms, structure_ids, energy,
                 measured, note=""):
        self.symbol = symbol
        self.binding = binding
        self.acronyms = tuple(acronyms)
        self.structure_ids = np.asarray(structure_ids, dtype=int)
        self.energy = np.asarray(energy, dtype=float)
        self.measured = np.asarray(measured, dtype=float)
        self.note = note
        if not (
            len(self.acronyms)
            == self.structure_ids.size
            == self.energy.size
            == self.measured.size
        ):
            raise ValueError("ragged comparison arrays")
        if self.energy.size < 3:
            raise ValueError(
                "%d regions is not a comparison; a rank correlation over "
                "fewer than three points is not a number worth printing"
                % self.energy.size
            )
        if not np.all(np.isfinite(self.energy)) or not np.all(
            np.isfinite(self.measured)
        ):
            raise ValueError("a NaN reached the comparison; filter first")
        if np.any(self.energy <= 0.0) or np.any(self.measured <= 0.0):
            raise ValueError(
                "a non-positive value reached the comparison; the log-log "
                "fit would be undefined and dropping it silently would be "
                "the wrong fix"
            )

    def __repr__(self):
        return "DensityComparison(%s vs %s, n=%d, rho=%+.3f)" % (
            self.symbol,
            self.binding.receptor,
            self.n,
            self.spearman[0],
        )

    @property
    def n(self):
        return int(self.energy.size)

    @property
    def spearman(self):
        """`(rho, p)` - does the map get the ordering right?"""
        result = stats.spearmanr(self.energy, self.measured)
        return float(result.statistic), float(result.pvalue)

    @property
    def pearson(self):
        """`(r, p)` on the raw values."""
        r, p = stats.pearsonr(self.energy, self.measured)
        return float(r), float(p)

    @property
    def log_pearson(self):
        """`(r, p)` on log values: the power-law fit's correlation."""
        r, p = stats.pearsonr(np.log(self.energy), np.log(self.measured))
        return float(r), float(p)

    @property
    def proportional_slope(self):
        """Least-squares `k_gene` for density = k * energy, through 0."""
        return float(
            (self.energy * self.measured).sum() / (self.energy ** 2).sum()
        )

    @property
    def proportional_r2(self):
        """Variance explained by the one-constant proportional model.

        Can be negative, and that is not a bug: a model with no free
        intercept can fit worse than the mean of the data.
        """
        predicted = self.proportional_slope * self.energy
        residual = ((self.measured - predicted) ** 2).sum()
        total = ((self.measured - self.measured.mean()) ** 2).sum()
        return float(1.0 - residual / total)

    @property
    def loglog(self):
        """`(slope, stderr, r2)` of log(density) against log(energy).

        Slope 1 is proportionality.
        """
        fit = stats.linregress(np.log(self.energy), np.log(self.measured))
        return float(fit.slope), float(fit.stderr), float(fit.rvalue ** 2)

    def predicted(self):
        """The proportional model's prediction, region by region."""
        return self.proportional_slope * self.energy

    def ratios(self):
        """measured / predicted, per region. 1.0 is a perfect prediction."""
        return self.measured / self.predicted()

    def worst(self, n=3):
        """The n regions the proportional model gets most wrong.

        Sorted by |log ratio|, so a 10x underprediction and a 10x
        overprediction rank equally badly.
        """
        ratios = self.ratios()
        order = np.argsort(np.abs(np.log(ratios)))[::-1][:n]
        return [
            (self.acronyms[i], float(self.measured[i]), float(self.predicted()[i]))
            for i in order
        ]

    def dynamic_ranges(self):
        """`(transcript max/min, measured max/min)` over these regions.

        The compression the log-log slope describes, in one pair of numbers.
        """
        return (
            float(self.energy.max() / self.energy.min()),
            float(self.measured.max() / self.measured.min()),
        )

    def describe(self):
        rho, p = self.spearman
        slope, stderr, r2 = self.loglog
        transcript_range, measured_range = self.dynamic_ranges()
        lines = [
            "%s vs %s %s (%s, %s)"
            % (
                self.symbol,
                self.binding.receptor,
                self.binding.ligand,
                self.binding.dataset.key,
                self.binding.dataset.species,
            ),
            "  n regions            %d  (%s)" % (self.n, ", ".join(self.acronyms)),
            "  Spearman rho         %+.3f  (p = %.4g)" % (rho, p),
            "  Pearson r            %+.3f" % self.pearson[0],
            "  log-log Pearson r    %+.3f" % self.log_pearson[0],
            "  proportional k       %.4g %s per energy unit"
            % (self.proportional_slope, self.binding.units),
            "  proportional R2      %+.3f" % self.proportional_r2,
            "  log-log slope        %.3f +/- %.3f (1.0 = proportional), R2 %.3f"
            % (slope, stderr, r2),
            "  dynamic range        transcript %.0fx, measured %.0fx"
            % (transcript_range, measured_range),
        ]
        for acronym, measured, predicted in self.worst():
            lines.append(
                "  worst: %-5s measured %8.1f, proportional model %8.1f (%.1fx)"
                % (acronym, measured, predicted, measured / predicted)
            )
        if self.note:
            lines.append("  note: %s" % self.note)
        return "\n".join(lines)


def compare(expression, symbol, binding, ontology=None, extra_regions=None):
    """Compare one gene's expression row against one published table.

    Joins on `autoradiography.REGION_MAPPING`, keeps only the regions that
    are in both the mapping and the expression matrix and that have a
    measured expression value, and reports what was dropped through
    `dropped_regions()` rather than silently.

    `extra_regions` is `{CCF acronym: published region name}` for joins the
    default mapping deliberately refuses to make - the substantia nigra,
    which CCF splits into SNc and SNr with no combined parent. Passing one
    is how a caller says which half they mean, in the open.
    """
    if ontology is None:
        ontology = expression.ontology
    if ontology is None:
        raise ValueError("an ontology is needed to join acronyms to structure ids")

    densities = binding.mapped(ontology=ontology)
    if extra_regions:
        for acronym, published in extra_regions.items():
            if published not in binding.values:
                raise KeyError(
                    "%r is not a region of this table" % (published,)
                )
            densities[acronym] = binding.values[published]

    row = expression.for_gene(symbol)
    ids = [int(i) for i in expression.structure_ids]

    acronyms, structure_ids, energies, measured = [], [], [], []
    for acronym in sorted(densities):
        structure_id = ontology.id_for_acronym(acronym)
        if structure_id not in ids:
            continue
        value = row[ids.index(structure_id)]
        if not np.isfinite(value) or value <= 0.0:
            continue
        acronyms.append(acronym)
        structure_ids.append(structure_id)
        energies.append(float(value))
        measured.append(float(densities[acronym]))

    return DensityComparison(
        symbol=symbol,
        binding=binding,
        acronyms=acronyms,
        structure_ids=structure_ids,
        energy=energies,
        measured=measured,
    )


def dropped_regions(expression, symbol, binding, ontology=None):
    """`[(what, why)]` for every published region not in a comparison.

    Four reasons a region can be missing, and they mean different things:
    no CCF counterpart (a join this module refuses to guess), not in the
    expression matrix (the caller asked for fewer regions), no ISH
    measurement (the twelve structures the Allen grid does not reach, plus
    per-experiment gaps), or an expression energy of exactly zero, which
    `compare()` drops because the log-log fit is undefined there and which
    would otherwise be an invisible exclusion.
    """
    if ontology is None:
        ontology = expression.ontology
    out = [
        (published, "no CCF counterpart: " + reason)
        for published, reason in binding.unmapped()
    ]
    row = expression.for_gene(symbol)
    ids = [int(i) for i in expression.structure_ids]
    for acronym, _ in sorted(binding.mapped().items()):
        structure_id = ontology.id_for_acronym(acronym)
        if structure_id not in ids:
            out.append((acronym, "not in this expression matrix"))
        elif not np.isfinite(row[ids.index(structure_id)]):
            out.append((acronym, "no Allen ISH measurement for %s" % symbol))
        elif row[ids.index(structure_id)] <= 0.0:
            out.append(
                (acronym, "%s expression energy is exactly zero here" % symbol)
            )
    return out


def fit_absolute_scale(comparison, note=""):
    """Fit `k_gene` from a comparison, with the fit quality attached.

    This is the only way an `AbsoluteScale` gets made, and it can only be
    made for a gene that has an independent measured density map in the
    same species. The R^2 travels with the constant because a k_gene fitted
    with R^2 = 0.34 is a real number and a bad one, and separating the two
    is how a bad number becomes a quoted number.
    """
    return AbsoluteScale(
        symbol=comparison.symbol,
        factor=comparison.proportional_slope,
        units=comparison.binding.units,
        n_regions=comparison.n,
        r_squared=comparison.proportional_r2,
        dataset=comparison.binding.dataset,
        ligand=comparison.binding.ligand,
        note=note,
    )


def compare_lothmann(expression, symbol, receptor, ontology=None):
    """`compare()` against one column of Lothmann et al. 2021."""
    binding = _auto.lothmann2021(receptor)
    if symbol not in binding.genes:
        raise ValueError(
            "%s is not one of the genes [3H]%s labels (%s). Comparing a "
            "gene against a ligand that does not bind its product would be "
            "a category error, not a weak result."
            % (symbol, binding.ligand, ", ".join(binding.genes) or "none")
        )
    return compare(expression, symbol, binding, ontology=ontology)


def policy_sensitivity(atlas, symbol, binding, structure_ids, ontology=None,
                       planes=None, combines=None, extra_regions=None):
    """The comparison's Spearman rho under every plane/combine policy.

    P4.1 refused to pick a plane policy or a combining rule and handed the
    choice to P4.2. This is P4.2 measuring what the choice is worth: if the
    answer moves a lot, the headline number needs its policy printed next to
    it, and if it barely moves, the refusal cost nothing.

    Returns `[(plane, combine, n, rho)]`, skipping the combinations that
    raise (`single` where a gene has several experiments, a plane a gene has
    no experiment in).
    """
    from .atlas import COMBINE_RULES, PLANE_POLICIES

    planes = list(planes or PLANE_POLICIES)
    combines = list(combines or COMBINE_RULES)
    out = []
    for plane in planes:
        for combine in combines:
            try:
                energy = atlas.energy_matrix(
                    structure_ids, symbols=[symbol], plane=plane, combine=combine
                )
                comparison = compare(
                    energy, symbol, binding, ontology=ontology,
                    extra_regions=extra_regions,
                )
            except (ValueError, KeyError):
                continue
            out.append((plane, combine, comparison.n, comparison.spearman[0]))
    return out
