#!/usr/bin/env python
"""Download and verify the Allen Mouse Brain Atlas ISH expression data this
project ingests.

Everything it knows about what to download lives in
`src/neurocascade/expression/sources.py`: query URLs, SHA-256 digests, sizes,
row counts, citations, and what the numbers mean. This script is only the
command line around it, so there is exactly one place a checksum is written
down.

    python scripts/fetch_expression.py                # fetch + verify all
    python scripts/fetch_expression.py --verify-only  # check the cache
    python scripts/fetch_expression.py --list         # what is in the manifest
    python scripts/fetch_expression.py --checksums    # sha256sum -c input
    python scripts/fetch_expression.py --missing      # genes with no ISH data
    python scripts/fetch_expression.py --gene Oprm1   # one gene, repeatable

Files land in `data/expression/`, which is gitignored: about 8 MB of
structure unionizes for 42 genes. The committed half of the contract is the
manifest, so a clean checkout plus this script reproduces the exact bytes or
fails saying which query moved.

Every entry here is a *live API query*, not an archive file, because Allen
does not publish the structure unionizes as a static dump. A digest mismatch
therefore has one extra legitimate cause compared with the connectome: Allen
added an ISH experiment, or re-ran the pipeline. That is still a change that
moves every downstream receptor map, so it is still a hard failure that a
human reads; `scripts/build_expression_manifest.py` regenerates the digest
table once someone has decided the change is wanted.

Exit status is 0 on success and 1 on any checksum mismatch or missing file,
so this is usable as a build step.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade.legacy.expression import fetch, genes, sources  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fetch and verify the Allen ISH expression source data."
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="where to cache (default: data/expression/ in this repo)",
    )
    parser.add_argument(
        "--key", action="append", dest="keys", help="fetch only this manifest key"
    )
    parser.add_argument(
        "--gene",
        action="append",
        dest="symbols",
        help="fetch only this gene's unionizes; repeatable",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="check the cache against the manifest without downloading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if a verified copy is cached",
    )
    parser.add_argument(
        "--list", action="store_true", help="print the manifest and exit"
    )
    parser.add_argument(
        "--checksums",
        action="store_true",
        help="print the manifest as `sha256sum -c` input and exit",
    )
    parser.add_argument(
        "--missing",
        action="store_true",
        help="print the searched-and-empty record for genes with no ISH data",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(argv)

    if args.checksums:
        for line in fetch.checksum_lines():
            print(line)
        return 0

    if args.missing:
        print("Genes in genes.RECEPTOR_GENES with NO usable Allen ISH data")
        print("(manifest %s, searched %s)" % (sources.MANIFEST_VERSION, "2026-08-12"))
        for entry in sources.GENES_WITHOUT_DATA:
            gene = genes.gene(entry["symbol"])
            print("")
            print("%s  (%s, Entrez %d)" % (gene.symbol, gene.protein, gene.entrez_id))
            print("  searched:   %s" % entry["searched"])
            print("  conclusion: %s" % entry["conclusion"])
        print("")
        print(
            "%d of %d genes have data; %d without."
            % (
                len(sources.AVAILABLE_SYMBOLS),
                len(genes.RECEPTOR_GENES),
                len(sources.GENES_WITHOUT_DATA),
            )
        )
        print("")
        print(
            "Sagittal-only (one hemisphere sectioned, weaker whole-brain "
            "coverage): %s" % ", ".join(sources.SAGITTAL_ONLY_SYMBOLS)
        )
        return 0

    if args.list:
        print("manifest version %s" % sources.MANIFEST_VERSION)
        for entry in sources.DATA_FILES:
            print("")
            print("%s" % entry.key)
            print("  file      %s (%d bytes)" % (entry.filename, entry.size_bytes))
            print("  sha256    %s" % entry.sha256)
            print("  url       %s" % entry.url)
            print("  retrieved %s (%s)" % (entry.retrieved, entry.volatility))
            if isinstance(entry, sources.UnionizeFile):
                print(
                    "  rows      %d from %d experiment(s)"
                    % (entry.n_rows, entry.n_section_data_sets)
                )
            print("  units     %s" % entry.units)
        return 0

    keys = list(args.keys or [])
    for symbol in args.symbols or []:
        keys.append(sources.unionize_key(symbol))
    if not keys:
        keys = [entry.key for entry in sources.DATA_FILES]
    elif sources.DATASET_INDEX_KEY not in keys:
        # A unionize file is unreadable without the index that says which
        # gene and plane each section_data_set_id is, so never fetch one
        # without the other.
        keys.insert(0, sources.DATASET_INDEX_KEY)

    cache_dir = args.cache_dir or fetch.DEFAULT_CACHE_DIR
    print("cache: %s" % cache_dir)
    failures = []

    for key in keys:
        entry = sources.data_file(key)
        try:
            if args.verify_only:
                fetch.verify(key, cache_dir=args.cache_dir)
                print("ok       %s  %s" % (entry.sha256[:16], entry.filename))
            else:
                path = fetch.download(
                    key,
                    cache_dir=args.cache_dir,
                    timeout=args.timeout,
                    force=args.force,
                )
                print(
                    "verified %s  %s (%d bytes)"
                    % (entry.sha256[:16], os.path.basename(path), entry.size_bytes)
                )
        except (fetch.ChecksumMismatch, fetch.MissingCachedFile) as error:
            failures.append((key, error))
            print("FAILED   %s" % key)
            print("         %s" % error)

    if failures:
        print("")
        print("%d of %d files failed." % (len(failures), len(keys)))
        print(sources.VOLATILITY.strip())
        return 1

    print("")
    print(
        "%d files verified against manifest %s."
        % (len(keys), sources.MANIFEST_VERSION)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
