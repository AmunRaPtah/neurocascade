"""P7.2: what a decoder nonlinearity changes, where one was adopted.

    python examples/nonlinear_decoder.py

Prints, for the endpoints P7.2 touched:

  * the model comparison that licensed the change;
  * the readout before and after, on the real connectome network;
  * the score the two decoders give across the adopted region's rate range,
    which is the whole point - a linear weight is a straight line through
    that range and the adopted basis is not.

Endpoints P7.2 left alone print one line saying so, because "we looked and
the data did not support a change" is the result for most of them.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np                              # noqa: E402

from neurocascade.legacy import connectome             # noqa: E402
from neurocascade import decoder                # noqa: E402
from neurocascade import endpoints              # noqa: E402
from neurocascade.legacy.connectome import fetch, sources   # noqa: E402


def main():
    adopted = decoder.adopted_nonlinearities()
    print("P7.2 adopted %d decoder nonlinearity(ies) from %d graded series."
          % (len(adopted), len(decoder.SERIES)))
    print()

    for verdict in decoder.all_endpoint_verdicts():
        if not verdict.series_verdicts:
            print("%-22s no graded single-region series; decoder unchanged"
                  % verdict.endpoint_key)
            continue
        print(verdict.report())
        print()

    network, ontology, source = _build()
    print("=" * 72)
    print("Readouts on %s" % source)
    print()

    for endpoint_key in sorted({s.endpoint for s in decoder.SERIES}):
        resolution = decoder.resolve_nonlinear(endpoint_key, network, ontology)
        print(resolution.report())
        print()

    if adopted:
        return 0

    print("=" * 72)
    print("COUNTERFACTUAL - nothing below was adopted")
    print("=" * 72)
    print("No series licensed a nonlinearity, so every readout above is")
    print("P7.1's. To show what an adoption would have changed, and to")
    print("exercise the machinery end to end, the inverted-U basis is")
    print("applied HERE ONLY to the node carrying the region P7.1 named as")
    print("P7.2's first target. This is a demonstration, not a result.")
    print()
    _counterfactual(network, ontology, "analgesia", "RM")
    return 0


def _counterfactual(network, ontology, endpoint_key, acronym):
    endpoint = endpoints.ENDPOINTS[endpoint_key]
    resolved = endpoint.resolve(network, ontology, include_contested=True)
    if acronym not in resolved.weights:
        print("  %s did not reach a node of this network." % acronym)
        return
    term = decoder.NonlinearTerm("U", licensed_by="NOTHING - demonstration")
    readout = decoder.NonlinearReadout(
        endpoint_key, resolved.weights, {acronym: term}
    )
    idx = network.index[acronym]
    print("  %s in %s, weight %+.4f, basis %s with peak_rate %.2f (%s)"
          % (acronym, endpoint_key, resolved.weights[acronym], term.shape,
             term.peak_rate, term.provenance))
    print("    %-8s %12s %12s" % ("rate", "P7.1 linear", "U basis"))
    for rate in np.linspace(0.0, 1.0, 11):
        rates = np.full(len(network.nodes), 0.5)
        rates[idx] = rate
        print("    %-8.2f %12.5f %12.5f"
              % (rate, resolved.score(network, rates),
                 readout.score(network, rates)))
    print()
    print("  The left column is a straight line in %s's rate by" % acronym)
    print("  construction; the right one turns round at 0.50. Deciding")
    print("  between them needs [ZhuoGebhart1997]'s four per-intensity")
    print("  means, which are in its Fig. 1A.")


def _build():
    """The real 291-region network when the cache is present, else fixtures."""
    cached = (
        fetch.is_cached("knox2019_connection_strength")
        and fetch.is_cached("knox2019_connection_density")
        and fetch.is_cached(sources.ONTOLOGY_KEY)
    )
    if cached:
        ontology = connectome.load_ontology()
        strength, voxels = connectome.load_knox_inputs(ontology=ontology)
        source = "Knox et al. 2019, 291 regions"
    else:
        ontology = connectome.load_fixture_ontology()
        strength, voxels = connectome.load_fixture_build_inputs(
            ontology=ontology
        )
        source = "the committed 15-region fixtures"
    circuit = connectome.build_circuit_network(
        strength, voxel_counts=voxels, resolution="regions"
    )
    return circuit.network, ontology, source


if __name__ == "__main__":
    sys.exit(main())
