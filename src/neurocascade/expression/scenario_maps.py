"""P4.3: the atlas-derived receptor density maps the scenarios consume.

Why this module exists at all
-----------------------------

P4.1 and P4.2 were both additive: nothing in `engine.py`, `receptor.py` or
`scenarios.py` imported the expression package, and every existing result
was unchanged. P4.3 is where that stops. Its third acceptance criterion is
to replace the hand-written multi-region mu-opioid density map with the
atlas-derived one, and this module is the join: it turns a gene symbol and
a `{node: CCF structure id}` mapping into the density factor
`ReceptorDensityMap.as_coupling()` needs, so `scenarios.py` gains three
lines rather than a data-loading path.

What is being replaced, precisely
---------------------------------

The old `mu_opioid_multi_region_coupling()` returned three hand-tuned
numbers, `{VTA_GABA: -3.0, PAG_Interneuron: -2.5, LC: -1.5}`. Its docstring
cited [TempelZukin1987] and [Mansour1987] for a *qualitative ordering* of
mu-receptor density (locus coeruleus and striatal patches high, PAG and VTA
moderate) and then said, correctly, that the numbers were not a
density-to-gain conversion: they folded receptor density together with how
much a unit of receptor activation moves that node given the circuit
around it.

P4.2 gave that folding a name and a factorization:

    coupling[node] = mechanism_gain[node] * rho[region]

so the replacement is exact rather than a rescaling. `rho` now comes from
the Allen Oprm1 ISH map instead of from a hand-read of two rat
autoradiography papers. `mechanism_gain` stays hand-set and stays labelled
illustrative, because no behavioural anchor for it exists (Track E and P2.2
both searched; `calibration.behavioral_evidence.SEARCHED_AND_EMPTY_
BEHAVIORAL` records the searches).

The normalization is `reference-region`, and that is load-bearing
----------------------------------------------------------------

`relative_density(normalization="reference-region", reference=VTA)` divides
the whole row by the VTA value, so VTA's rho is 1.0 by construction and the
other two regions are expressed as multiples of it. Three consequences,
each of which is why this policy and not `max` or `mean`:

- **VTA_GABA's coupling is unchanged.** It is the node the single-region
  scenario also uses, its gain is the one P2.2 fitted a posterior for, and
  a replacement that moved it would have changed a calibrated number under
  cover of a data-ingestion task.
- **The result does not depend on which other regions are in the matrix.**
  Dividing by one named region is invariant to the rest of the row, where
  dividing by the peak or the mean is not. That is what makes the number
  the same whether it is computed from the 291-region connectome matrix,
  the 69-structure committed fixture, or the three regions alone - and
  `tests/test_scenario_maps.py` asserts all three agree.
- **It is the case `density.NORMALIZATIONS` documents for exactly this
  situation**: "use when a scenario has a region whose coupling gain is
  already calibrated and the others should be expressed against it".

Which atlas, and why the fixture is not a shortcut here
-------------------------------------------------------

`DEFAULT_SOURCE = "fixture"`. The expression fixtures are byte-identical
slices of the checksum-verified Allen responses (P4.1 cut them through
`csv.reader`/`csv.writer` without ever parsing a float, and a `real_data`
test asserts the arrays match over every structure of every experiment), so
for a quantity that depends only on the three regions' own rows, "from the
fixture" and "from the full file" are the same number, not an approximation
of it. `expression/fixtures/__init__.py` warns that 69 structures are a
test input and not a receptor map, and that warning is about maps whose
value depends on the region set - a peak-normalized map, a coverage
statistic, a correlation. This one does not.

The alternative was to make `scenarios.py` require the gitignored 7.5 MB
cache, which would have made every existing scenario test fail in a clean
checkout. `source="cache"` does that on request, and the equality is
tested rather than asserted in prose.
"""

from . import density as _density

#: Where `load()` reads from by default. See the module docstring.
SOURCES = ("fixture", "cache")
DEFAULT_SOURCE = "fixture"

#: The plane and combine policy the scenario maps use. Coronal experiments
#: cover the whole brain; `coronal-else-sagittal` never pools two
#: geometries for one gene. P4.1 refused to default these and P4.2 measured
#: the refusal to be worth little for orderings; `policy_spread()` below
#: measures it again for exactly these three regions rather than assuming
#: the earlier measurement carries.
PLANE = "coronal-else-sagittal"
COMBINE = "mean"


def load_atlas(symbols, source=DEFAULT_SOURCE, ontology=None, cache_dir=None):
    """The expression atlas a scenario map is built from."""
    if source not in SOURCES:
        raise ValueError("source must be one of %s, got %r" % (list(SOURCES), source))
    from .. import connectome

    if source == "fixture":
        from . import fixtures

        if ontology is None:
            ontology = connectome.load_fixture_ontology()
        return fixtures.load_fixture_atlas(symbols=list(symbols), ontology=ontology)

    from .atlas import load_expression_atlas

    if ontology is None:
        ontology = connectome.load_ontology()
    return load_expression_atlas(
        symbols=list(symbols), ontology=ontology, cache_dir=cache_dir
    )


def scenario_density_map(
    symbol,
    nodes,
    reference_node,
    source=DEFAULT_SOURCE,
    ontology=None,
    atlas=None,
    plane=PLANE,
    combine=COMBINE,
    acknowledge_mismatch=False,
):
    """`ReceptorDensityMap` over the CCF structures a scenario's nodes stand for.

    `nodes` is `{node name: CCF structure id}` and `reference_node` names
    the node whose region becomes 1.0. Two nodes may share a region (the
    VTA scenario has `VTA_GABA` and `VTA_DA` on one structure); this
    function is given the structures, and the caller decides which nodes
    carry a receptor.
    """
    if reference_node not in nodes:
        raise KeyError(
            "reference_node %r is not one of the nodes %s"
            % (reference_node, sorted(nodes))
        )
    if atlas is None:
        atlas = load_atlas([symbol], source=source, ontology=ontology)
    structure_ids = sorted(set(int(i) for i in nodes.values()))
    expression = atlas.energy_matrix(
        structure_ids, symbols=[symbol], plane=plane, combine=combine
    )
    return _density.relative_density(
        expression,
        symbol,
        normalization="reference-region",
        reference=int(nodes[reference_node]),
        acknowledge_mismatch=acknowledge_mismatch,
    )


def scenario_coupling(
    symbol,
    nodes,
    mechanism_gains,
    reference_node,
    source=DEFAULT_SOURCE,
    ontology=None,
    atlas=None,
    plane=PLANE,
    combine=COMBINE,
    acknowledge_mismatch=False,
):
    """`{node: mechanism_gain * rho}` - the coupling dict `Simulation` takes.

    The multiplication is `ReceptorDensityMap.as_coupling()`'s, so the
    factorization P4.2 defined is the one actually executed here rather
    than a second implementation of it.
    """
    density_map = scenario_density_map(
        symbol,
        nodes,
        reference_node,
        source=source,
        ontology=ontology,
        atlas=atlas,
        plane=plane,
        combine=combine,
        acknowledge_mismatch=acknowledge_mismatch,
    )
    return density_map.as_coupling(mechanism_gains, nodes)


def policy_spread(symbol, nodes, reference_node, source=DEFAULT_SOURCE,
                  ontology=None, atlas=None):
    """`{(plane, combine): {node: rho}}` over every policy that has data.

    The loader's two deferred choices, priced for this specific map. A
    policy that changes the ordering of a scenario's three regions would be
    a modelling decision hiding inside a default, which is the thing P4.1
    built the required keyword arguments to prevent.
    """
    from .atlas import COMBINE_RULES, PLANE_POLICIES

    if atlas is None:
        atlas = load_atlas([symbol], source=source, ontology=ontology)
    out = {}
    for plane in PLANE_POLICIES:
        for combine in COMBINE_RULES:
            try:
                density_map = scenario_density_map(
                    symbol, nodes, reference_node, atlas=atlas,
                    plane=plane, combine=combine, acknowledge_mismatch=True,
                )
            except ValueError:
                # No experiment in this plane, or combine="single" with
                # several: a policy with no data is not a policy whose
                # answer disagrees, and conflating the two would be the
                # same mistake as reading NaN as zero.
                continue
            out[(plane, combine)] = {
                node: density_map.for_region(structure_id)
                for node, structure_id in nodes.items()
            }
    return out
