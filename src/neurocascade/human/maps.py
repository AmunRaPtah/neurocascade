"""The human receptor-maps layer: expression -> the coupling dict the engine
consumes.

The human analogue of `expression/scenario_maps.py`. `human/expression.py`
loads one gene's AHBA row and converts it to a relative density
(`relative_density`), and `HumanReceptorDensityMap.as_coupling()` does the
`gain * rho` multiplication that `engine.Simulation` reads. This module is
the JOIN that runs that over the whole 82-region brain: it builds
`{node: gain * rho[region]}` for every region the expression map measures,
and it is the one place the question "what happens to a region the map does
NOT measure" is answered for the human data rather than answered implicitly.

The question is not hypothetical. Two Desikan-Killiany regions - the right
frontal pole and the right temporal pole (`R_frontalpole`, `R_temporalpole`)
- were matched to no AHBA tissue sample, so their rows are all-NaN in the
published file (`human/expression.REGIONS_WITHOUT_SAMPLES`). A whole-brain
receptor map must decide what to do with them, and the three possible
answers are not equivalent:

* `missing="raise"`  (the default) refuses to build a coupling dict that
                     would otherwise have to invent a density for those two
                     regions. This is the same refusal
                     `as_coupling()` makes, restated at whole-brain scale
                     with every offending region named at once.
* `missing="exclude"` drops the two regions from the coupling dict. The
                     engine then simply adds no receptor drive at those two
                     nodes - which is the honest reading: the map does not
                     reach them, so the model does not place a receptor
                     there. It is NOT the claim "the receptor is absent
                     there", because the two paths are distinguished by the
                     `excluded` field the result carries.
* substituting zero for them is deliberately NOT offered. Zero is the claim
  "a drug acting at this receptor has no target in this region", and "not
  measured" is not that claim (`human/expression.py` says this about NaN
  and `maps` inherits it rather than re-deciding it).

Everything else in this module is the same factorization the mouse package
uses, stated once more because it is the part a reader most often gets
backwards:

    coupling[node] = mechanism_gain[node] * rho[region]

`rho` is `relative_density()`'s dimensionless per-region relative density
(from AHBA expression, under the proportionality ASSUMPTION stated in
`human/expression.py`); `mechanism_gain` is the OTHER factor - how much a
unit of receptor activation moves that node given the circuit around it -
and it is NOT this layer's to supply. No `mechanism_gain` for a human node
has ever been fitted in this repository, so any caller that passes one is
asserting a gain whose provenance is the caller's, exactly as
`expression/scenario_maps.py` documents for the mouse scenarios.

    from neurocascade import human

    expr = human.load_expression()
    coupling = human.whole_brain_coupling(
        expr, "DRD2", mechanism_gains=-3.0, normalization="max",
        missing="exclude",
    )
    print(coupling.summary())
    print(coupling.coupling)      # the dict Simulation(coupling=...) takes
"""

import numpy as np

from . import expression as _expression

__all__ = [
    "MISSING_POLICIES",
    "HumanCoupling",
    "whole_brain_nodes",
    "unmeasured_regions",
    "whole_brain_coupling",
]

#: What to do with a region the expression map does not measure. There is
#: deliberately no "zero" option: see the module docstring for why zero is
#: the one reading that must not be available.
MISSING_POLICIES = ("raise", "exclude")


class HumanCoupling(object):
    """One gene's whole-brain coupling dict, and which regions were dropped.

    `coupling` is the `{node: gain * rho[region]}` dict `Simulation` takes.
    `excluded` records the regions dropped under `missing="exclude"`, so the
    two facts - "no receptor drive at this node" and "the receptor is absent
    there" - cannot be read back as the same thing later.
    """

    def __init__(self, coupling, symbol, normalization, reference, excluded,
                 missing):
        self.coupling = dict(coupling)
        self.symbol = symbol
        self.normalization = normalization
        self.reference = reference
        self.excluded = tuple(excluded)
        self.missing = missing

    @property
    def n_nodes(self):
        return len(self.coupling)

    def gain_for(self, node):
        return self.coupling[node]

    def summary(self):
        lines = [
            "%s whole-brain coupling (normalization=%r%s)"
            % (
                self.symbol,
                self.normalization,
                "" if self.reference is None else ", reference=%r" % self.reference,
            ),
            "  %d nodes carry this receptor" % self.n_nodes,
        ]
        if self.excluded:
            lines.append(
                "  excluded (no measured expression, so no receptor placed): %s"
                % ", ".join(self.excluded)
            )
        else:
            lines.append("  excluded: none")
        return "\n".join(lines)

    def __repr__(self):
        return "HumanCoupling(%r, %d nodes)" % (self.symbol, self.n_nodes)


def whole_brain_nodes(labels):
    """`{label: label}`: every human node IS one Desikan-Killiany region.

    Unlike the mouse scenarios, where several populations can share one CCF
    region, the human build is one node per atlas region (`human/build.py`),
    so the node name and the region label are the same string.
    """
    return {label: label for label in labels}


def unmeasured_regions(expression, symbol, labels=None):
    """Region labels whose row for `symbol` is NaN (no AHBA sample matched).

    `labels` defaults to the expression matrix's own region order. NaN is
    not zero: these are the regions `relative_density()` deliberately keeps
    as NaN and `as_coupling()` deliberately refuses to emit a factor for.
    """
    if labels is None:
        labels = list(expression.region_labels)
    labels = list(labels)
    row = expression.for_gene(symbol)
    index = {label: i for i, label in enumerate(expression.region_labels)}
    return [label for label in labels if not np.isfinite(row[index[label]])]


def whole_brain_coupling(
    expression,
    symbol,
    mechanism_gains,
    labels=None,
    normalization="max",
    reference=None,
    missing="raise",
):
    """`HumanCoupling` for one gene over the whole 82-region human brain.

    `mechanism_gains` is a single number (applied to every node) or a
    per-node dict; it is the factor this layer does not supply, and every
    caller that passes a number is asserting that number's provenance. The
    multiplication is `HumanReceptorDensityMap.as_coupling()`'s, so the
    factorization is executed here rather than re-implemented.

    `labels` defaults to the expression matrix's region order, which is the
    human connectome's (`human.load_connectome().labels`), so node names
    line up with the network `human/build.py` builds.

    `missing` is `"raise"` (refuse to build with unmeasured regions) or
    `"exclude"` (drop them and record them in `HumanCoupling.excluded`).
    """
    if missing not in MISSING_POLICIES:
        raise ValueError(
            "missing must be one of %s, got %r. There is no 'zero' policy: "
            "substituting zero would assert the receptor is absent in an "
            "unmeasured region, and 'not measured' is not that claim."
            % (list(MISSING_POLICIES), missing)
        )
    if labels is None:
        labels = list(expression.region_labels)
    labels = list(labels)
    if not labels:
        raise ValueError("labels is empty; there is nothing to build a map over")

    density = _expression.relative_density(
        expression,
        symbol,
        normalization=normalization,
        reference=reference,
    )
    excluded = unmeasured_regions(expression, symbol, labels)
    nodes = whole_brain_nodes(labels)
    gains = mechanism_gains

    if excluded:
        if missing == "raise":
            raise ValueError(
                "%s has no measured expression in %s, so these nodes have "
                "no density factor. missing='exclude' drops them from the "
                "coupling dict (the engine then adds no receptor drive at "
                "those nodes); substituting zero would assert the receptor "
                "is absent there, and 'not measured' is not that claim."
                % (symbol, ", ".join(excluded))
            )
        dropped = set(excluded)
        kept = [label for label in labels if label not in dropped]
        nodes = whole_brain_nodes(kept)
        if isinstance(gains, dict):
            gains = {node: g for node, g in gains.items() if node not in dropped}

    coupling = density.as_coupling(gains, nodes)
    return HumanCoupling(
        coupling=coupling,
        symbol=symbol,
        normalization=normalization,
        reference=reference,
        excluded=excluded,
        missing=missing,
    )
