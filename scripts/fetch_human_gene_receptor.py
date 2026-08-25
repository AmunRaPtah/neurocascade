#!/usr/bin/env python
"""Download and verify the gene-receptor correspondence table (P12.4).

Everything it knows about what to download lives in
`src/neurocascade/human/gene_receptor.py`: the URL (pinned to one commit of
`netneurolab/hansen_gene-receptor`), the SHA-256, the size, the citations and
licence, and what the columns mean. This script is only the command line
around it, so there is exactly one place a checksum is written down.

    python scripts/fetch_human_gene_receptor.py              # fetch + verify
    python scripts/fetch_human_gene_receptor.py --verify-only
    python scripts/fetch_human_gene_receptor.py --list       # manifest
    python scripts/fetch_human_gene_receptor.py --checksums  # sha256sum -c

The file lands in `data/human/` next to the PET atlas files (4 KB). It is
the measured reason the ISH-expression-to-density path was retired into
`legacy/`: its GABA-A rows are the numbers `tests/test_human_gene_receptor.py`
asserts and `docs/heldout-failure-analysis.md` quotes.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade.human import gene_receptor  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Fetch and verify the Hansen 2022 gene-expression/receptor-"
            "density correspondence table."
        )
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="where to cache (default: data/human/ in this repo)",
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
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(argv)

    if args.checksums:
        for line in gene_receptor.checksum_lines():
            print(line)
        return 0

    if args.list:
        print("manifest version %s" % gene_receptor.MANIFEST_VERSION)
        print("source commit %s" % gene_receptor.REPOSITORY_COMMIT)
        for entry in gene_receptor.DATA_FILES:
            print("")
            print("%s" % entry.key)
            print("  file      %s (%d bytes)" % (entry.filename, entry.size_bytes))
            print("  sha256    %s" % entry.sha256)
            print("  url       %s" % entry.url)
            print("  retrieved %s (%s)" % (entry.retrieved, entry.volatility))
            print("  units     %s" % entry.units)
            print("  source    %s" % entry.citation)
        return 0

    cache_dir = args.cache_dir or gene_receptor.DEFAULT_CACHE_DIR
    print("cache: %s\n" % cache_dir)
    failures = 0
    try:
        for entry in gene_receptor.DATA_FILES:
            if args.verify_only:
                path = gene_receptor.verify(entry.key, cache_dir=cache_dir)
                print("ok       %s  %s" % (entry.sha256[:16], entry.filename))
            else:
                path = gene_receptor.download(
                    entry.key, cache_dir=cache_dir, timeout=args.timeout,
                    force=args.force,
                )
                print(
                    "verified %s  %s (%d bytes)"
                    % (
                        entry.sha256[:16],
                        os.path.basename(path),
                        entry.size_bytes,
                    )
                )
    except (
        gene_receptor.ChecksumMismatch,
        gene_receptor.MissingCachedFile,
    ) as error:
        print("FAILED")
        print("      %s" % error)
        failures += 1

    if failures:
        return 1
    print("")
    print("all %d file(s) verified against manifest %s."
          % (len(gene_receptor.DATA_FILES), gene_receptor.MANIFEST_VERSION))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
