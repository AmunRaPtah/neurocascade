#!/usr/bin/env python
"""Download and verify the human AHBA gene-expression data this project ingests.

Everything it knows about what to download lives in
`src/neurocascade/human/expression.py`: the URL (pinned to a commit), the
SHA-256 digest, the size, citations, and what the matrix's numbers mean.
This script is only the command line around it, so there is exactly one
place a checksum is written down.

    python scripts/fetch_human_expression.py              # fetch + verify
    python scripts/fetch_human_expression.py --verify-only
    python scripts/fetch_human_expression.py --list       # manifest
    python scripts/fetch_human_expression.py --checksums  # sha256sum -c input

The file lands in `data/human/` next to the connectome files (about 19.6 MB).
The committed half of the contract is the manifest, so a clean checkout plus
this script reproduces the exact bytes or fails saying what moved.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade.human import expression  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fetch and verify the human AHBA expression source data."
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
        for line in expression.checksum_lines():
            print(line)
        return 0

    if args.list:
        print("manifest version %s" % expression.MANIFEST_VERSION)
        for entry in expression.DATA_FILES:
            print("")
            print("%s" % entry.key)
            print("  file      %s (%d bytes)" % (entry.filename, entry.size_bytes))
            print("  sha256    %s" % entry.sha256)
            print("  url       %s" % entry.url)
            print("  retrieved %s (%s)" % (entry.retrieved, entry.volatility))
            print("  units     %s" % entry.units)
            print("  source    %s" % entry.citation)
        return 0

    cache_dir = args.cache_dir or expression.DEFAULT_CACHE_DIR
    print("cache: %s" % cache_dir)
    entry = expression.data_file(expression.DEFAULT_EXPRESSION_KEY)
    try:
        if args.verify_only:
            path = expression.verify(
                expression.DEFAULT_EXPRESSION_KEY, cache_dir=args.cache_dir
            )
            print("ok       %s  %s" % (entry.sha256[:16], entry.filename))
        else:
            path = expression.download(
                expression.DEFAULT_EXPRESSION_KEY,
                cache_dir=args.cache_dir,
                timeout=args.timeout,
                force=args.force,
            )
            print(
                "verified %s  %s (%d bytes)"
                % (entry.sha256[:16], os.path.basename(path), entry.size_bytes)
            )
    except (expression.ChecksumMismatch, expression.MissingCachedFile) as error:
        print("FAILED")
        print("      %s" % error)
        return 1

    print("")
    print("file verified against manifest %s." % expression.MANIFEST_VERSION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
