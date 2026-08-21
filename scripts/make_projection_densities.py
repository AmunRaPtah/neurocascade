#!/usr/bin/env python
"""Regenerate the committed P6.3 projection-density table from the cache.

The table is DERIVED DATA, not a new source: every number in it is

    sum_{i in nucleus} sum_{j in region} connection_strength(i, j)
      / sum_{j in region} voxel_count(j)

over the checksum-verified Knox et al. 2019 matrices that P3.1 already
ingested, with the voxel counts recovered from those same matrices by
`connectome/volumes.py`. It is committed for the reason the connectome
fixtures are committed: the offline test gate cannot read the gitignored
8 MB cache, and 162 numbers can be reviewed in a diff.

`fetch.verify()` runs on both input files before either is read, so the
table can never be cut from unverified bytes -- the same guarantee
`make_connectome_fixture.py` gives.

Usage:  python scripts/make_projection_densities.py [--cache-dir DIR] [--check]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade.legacy import projection_density as pdens  # noqa: E402
from neurocascade.legacy.connectome import fetch, sources  # noqa: E402
from neurocascade.legacy.connectome import volumes  # noqa: E402
from neurocascade.legacy.connectome.matrix import load_connectome  # noqa: E402
from neurocascade.legacy.connectome.ontology import load_ontology  # noqa: E402

INPUT_KEYS = ("knox2019_connection_strength", "knox2019_connection_density")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--check",
        action="store_true",
        help="derive and compare against the committed table without writing",
    )
    args = parser.parse_args(argv)

    for key in (sources.ONTOLOGY_KEY,) + INPUT_KEYS:
        fetch.verify(key, cache_dir=args.cache_dir)
    print("verified: %s" % ", ".join((sources.ONTOLOGY_KEY,) + INPUT_KEYS))

    ontology = load_ontology(cache_dir=args.cache_dir)
    strength = load_connectome(
        key=INPUT_KEYS[0], ontology=ontology, cache_dir=args.cache_dir
    )
    density = load_connectome(
        key=INPUT_KEYS[1], ontology=ontology, cache_dir=args.cache_dir
    )
    voxels = volumes.target_voxel_counts(strength, density)
    derived = pdens.derive_table(strength, voxels, ontology)

    if args.check:
        committed = pdens.load_table()
        worst = 0.0
        for (nucleus, region), value in sorted(derived.items()):
            other = committed.density(region, nucleus)
            if value != other:
                worst = max(worst, abs(value - other))
                print("MISMATCH %s -> %s: %r vs %r" % (nucleus, region, value, other))
        print("check: worst absolute difference %g" % worst)
        return 0 if worst == 0.0 else 1

    header = [
        "P6.3 projection densities, DERIVED from the Allen Mouse Brain",
        "Connectivity Atlas. Regenerate with scripts/make_projection_densities.py.",
        "",
        "rho(nucleus -> region) = sum_{i in nucleus} sum_{j in region}",
        "    connection_strength(i, j) / sum_{j in region} voxel_count(j)",
        "",
        "units: %s" % pdens.TABLE_UNITS,
        "",
        "ipsilateral block only, as everywhere else in this package.",
        "",
    ]
    for key in INPUT_KEYS:
        entry = sources.data_file(key)
        header.append("%s sha256 %s" % (key, entry.sha256))
    header.append(
        "voxel counts recovered from those two files by connectome/volumes.py"
    )
    header.append("")
    header.append(
        "kind=nuclear rows are RECORDED AND REFUSED; see projection_density.py."
    )

    pdens.write_table(
        pdens.TABLE_PATH,
        derived,
        dataset_key=strength.dataset_key,
        citation=strength.citation,
        units=pdens.TABLE_UNITS,
        header_lines=header,
    )
    print("wrote %s (%d rows)" % (pdens.TABLE_PATH, len(derived)))
    table = pdens.load_table()
    print()
    print(table.report())
    print()
    print(table.validation_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
