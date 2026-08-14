"""P7.1: dump the behavioural endpoint library and how it resolves.

    python scripts/report_endpoints.py > results/behavioral_endpoints.txt

Prints, for every endpoint: its assay, its polarity, the cited term behind
every region, and then the resolved weights at each of the three connectome
resolutions, so that "which regions survive a coarse parcellation" is a
number a reader can see rather than a claim.

Runs off the gitignored data/connectome cache; falls back to the committed
fifteen-region fixtures when the cache is absent, and says which it used.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade import connectome, endpoints                # noqa: E402
from neurocascade.connectome import build, fetch, sources     # noqa: E402


def load():
    cached = (
        fetch.is_cached("knox2019_connection_strength")
        and fetch.is_cached("knox2019_connection_density")
        and fetch.is_cached(sources.ONTOLOGY_KEY)
    )
    if cached:
        ontology = connectome.load_ontology()
        strength, voxels = connectome.load_knox_inputs(ontology=ontology)
        return (ontology, strength, voxels,
                "the full published Knox et al. 2019 matrices")
    ontology = connectome.load_fixture_ontology()
    strength, voxels = connectome.load_fixture_build_inputs(ontology=ontology)
    return (ontology, strength, voxels,
            "the committed fifteen-region fixtures")


def node_sizes(ontology, circuit, voxel_counts):
    """{node acronym: voxel count}, summed over the fine regions in it.

    Only used to split a term across descendant nodes in proportion to their
    volume, which is the arithmetic a parent region's mean rate implies.
    """
    report = circuit.report
    _, mapping = build.parcellation(
        ontology, report.fine_region_ids, report.resolution
    )
    totals = {i: 0.0 for i in report.region_ids}
    for fine in report.fine_region_ids:
        # `volumes.target_voxel_counts` is keyed by (hemisphere, id); this
        # is one hemisphere, matching `build._coarse_target_voxels`.
        totals[mapping[fine]] += float(voxel_counts[("ipsi", int(fine))])
    return {
        ontology[i].acronym: totals[i] for i in report.region_ids
    }


def main():
    ontology, strength, voxels, source = load()
    print("Behavioural endpoint library (P7.1)")
    print("=" * 72)
    print("source: %s" % source)
    print("weights: %s -- every magnitude is illustrative until P7.3 fits it"
          % endpoints.ILLUSTRATIVE_WEIGHTS)
    print()

    circuits = {}
    for resolution in build.RESOLUTIONS:
        circuit = connectome.build_circuit_network(
            strength, voxel_counts=voxels, resolution=resolution
        )
        circuits[resolution] = (circuit, node_sizes(ontology, circuit, voxels))

    for key in endpoints.ENDPOINT_ORDER:
        endpoint = endpoints.ENDPOINTS[key]
        print(endpoint.summary())
        print()
        for resolution in build.RESOLUTIONS:
            circuit, sizes = circuits[resolution]
            resolved = endpoint.resolve(
                circuit.network, ontology, region_sizes=sizes
            )
            print("  resolution=%s (%d nodes)" % (resolution, circuit.n))
            for line in resolved.report().splitlines()[1:]:
                print("  " + line)
            print("  score at rest: %+.4f" % resolved.baseline_score())
            print()
        print("-" * 72)

    print()
    print("Citations")
    print("=" * 72)
    for citation in sorted(endpoints.CITATIONS):
        print("[%s] %s" % (citation, endpoints.CITATIONS[citation]))
        print()
    print(endpoints.SEARCHED_AND_EMPTY)


if __name__ == "__main__":
    main()
