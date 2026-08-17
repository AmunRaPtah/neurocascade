#!/usr/bin/env python
"""Download and verify the human connectome data this project ingests.

Everything it knows about what to download lives in
`src/neurocascade/human/sources.py`: URLs (pinned to a commit), SHA-256
digests, sizes, citations, and what the matrix's numbers mean. This script
is only the command line around it, so there is exactly one place a
checksum is written down.

    python scripts/fetch_human_connectome.py                # fetch + verify all
    python scripts/fetch_human_connectome.py --verify-only  # check the cache
    python scripts/fetch_human_connectome.py --list         # what is in the manifest
    python scripts/fetch_human_connectome.py --checksums    # sha256sum -c input

Files land in `data/human/`, which is gitignored (about 26 KB). The
committed half of the contract is the manifest, so a clean checkout plus
this script reproduces the exact bytes or fails saying which file moved.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade.human import fetch, sources  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fetch and verify the human connectome source data."
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="where to cache (default: data/human/ in this repo)",
    )
    parser.add_argument(
        "--key",
        action="append",
        dest="keys",
        help="fetch only this manifest key; repeatable",
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
        for line in fetch.checksum_lines():
            print(line)
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
            print("  units     %s" % entry.units)
            print("  source    %s" % entry.citation)
        return 0

    keys = args.keys or [entry.key for entry in sources.DATA_FILES]
    cache_dir = args.cache_dir or fetch.DEFAULT_CACHE_DIR
    print("cache: %s" % cache_dir)
    failures = []

    for key in keys:
        entry = sources.data_file(key)
        try:
            if args.verify_only:
                path = fetch.verify(key, cache_dir=args.cache_dir)
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
    print("%d files verified against manifest %s." % (len(keys), sources.MANIFEST_VERSION))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
