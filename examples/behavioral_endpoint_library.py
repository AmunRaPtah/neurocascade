"""P7.1: the seven behavioural endpoints, read out over the whole brain.

    python examples/behavioral_endpoint_library.py

What this shows, and what it deliberately does not
--------------------------------------------------

It shows the three things P7.1 delivers: that each endpoint is a readout
over several Allen-CCF regions, that each region is there because of a
cited experiment, and that the same seven endpoints resolve onto a
whole-brain network at any of the three connectome resolutions.

It does NOT predict a drug effect. Every weight magnitude here is
illustrative and equal (`endpoints.ILLUSTRATIVE_WEIGHTS`), so a score
change is a statement about which regions moved and in which direction,
not about how many centimetres an animal walked. Fitting the magnitudes
against real dose-response curves is P7.3.

The perturbation below is therefore a probe, not a simulation: one region
is nudged by hand and the seven endpoints are read, which is the cleanest
way to see the readout's polarity and its cross-endpoint structure. Note
in particular that raising VTA raises locomotion, raises reward AND lowers
sedation, from three independent sets of citations that were never fitted
to agree.

Falls back to the committed fifteen-region fixtures when the gitignored
data/connectome cache is absent, and says which it used.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np                                            # noqa: E402

from neurocascade import connectome, endpoints                # noqa: E402
from neurocascade.connectome import fetch, sources            # noqa: E402


PROBES = ("VTA", "SNr", "PAG", "LC", "VLPO", "BLA")


def load():
    cached = (
        fetch.is_cached("knox2019_connection_strength")
        and fetch.is_cached("knox2019_connection_density")
        and fetch.is_cached(sources.ONTOLOGY_KEY)
    )
    if cached:
        ontology = connectome.load_ontology()
        strength, voxels = connectome.load_knox_inputs(ontology=ontology)
        return ontology, strength, voxels, "Knox et al. 2019, 291 regions"
    ontology = connectome.load_fixture_ontology()
    strength, voxels = connectome.load_fixture_build_inputs(ontology=ontology)
    return ontology, strength, voxels, "the committed 15-region fixtures"


def main():
    ontology, strength, voxels, source = load()
    circuit = connectome.build_circuit_network(
        strength, voxel_counts=voxels, resolution="regions"
    )
    network = circuit.network
    print("Behavioural endpoint library, on %s" % source)
    print("=" * 70)
    print()

    resolved = {}
    for key in endpoints.ENDPOINT_ORDER:
        endpoint = endpoints.ENDPOINTS[key]
        r = endpoint.resolve(network, ontology)
        resolved[key] = r
        print(
            "%-20s %2d region(s), %2d cited term(s), %d contested and excluded"
            % (
                key,
                len(r.weights),
                len(endpoint.active_terms()),
                len(endpoint.terms) - len(endpoint.active_terms()),
            )
        )
    print()

    print("Score at rest (every node at 0.5), and the change when one")
    print("region is raised by 0.2 from rest:")
    print()
    header = "%-14s" % "probe"
    for key in endpoints.ENDPOINT_ORDER:
        header += "%12s" % key[:11]
    print(header)
    rest = np.full(network.n, 0.5)
    line = "%-14s" % "(rest)"
    for key in endpoints.ENDPOINT_ORDER:
        line += "%12.4f" % resolved[key].score(network, rest)
    print(line)
    print("-" * len(header))

    for probe in PROBES:
        if probe not in network.index:
            print("%-14s (not a node of this network)" % probe)
            continue
        rates = rest.copy()
        rates[network.index[probe]] += 0.2
        line = "%-14s" % ("+0.2 " + probe)
        for key in endpoints.ENDPOINT_ORDER:
            delta = resolved[key].delta(network, rates)
            line += "%+12.4f" % (0.0 if delta == 0.0 else delta)
        print(line)

    print()
    print("A zero above means the region carries no cited term for that")
    print("endpoint, which is a statement about the literature and not")
    print("about the brain; endpoints.SEARCHED_AND_EMPTY records the gaps.")
    print()
    print("The two assay-restricted views of analgesia, which exist because")
    print("two papers ran both assays and found them to dissociate:")
    for key, variant in endpoints.ASSAY_VARIANTS.items():
        r = variant.resolve(network, ontology)
        print("  %-24s %s" % (key, ", ".join(sorted(r.weights))))


if __name__ == "__main__":
    main()
