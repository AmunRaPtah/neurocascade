#!/usr/bin/env python
"""Download and verify the human PET receptor-atlas data this project ingests.

Everything it knows about what to download lives in
`src/neurocascade/human/receptors.py`: the URLs (pinned to a commit), the
SHA-256 digests, the sizes, the citations and licences, and what each file's
numbers mean. This script is only the command line around it, so there is
exactly one place a checksum is written down.

    python scripts/fetch_human_receptors.py              # fetch + verify all 7
    python scripts/fetch_human_receptors.py --verify-only
    python scripts/fetch_human_receptors.py --list       # manifest
    python scripts/fetch_human_receptors.py --checksums  # sha256sum -c input

Files land in `data/human/` next to the connectome files (about 100 KB
total: five parcellated tracer maps, the Cammoun-033 parcel-identity xlsx,
and the paper's deposited aggregation matrix). The committed half of the
contract is the manifest, so a clean checkout plus this script reproduces
the exact bytes or fails saying what moved.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade.human import receptors  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fetch and verify the Hansen 2022 human PET receptor atlas."
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
        for line in receptors.checksum_lines():
            print(line)
        return 0

    if args.list:
        print("manifest version %s" % receptors.MANIFEST_VERSION)
        print("source commit %s" % receptors.REPOSITORY_COMMIT)
        for entry in receptors.DATA_FILES:
            print("")
            print("%s" % entry.key)
            print("  file      %s (%d bytes)" % (entry.filename, entry.size_bytes))
            print("  sha256    %s" % entry.sha256)
            print("  url       %s" % entry.url)
            print("  retrieved %s (%s)" % (entry.retrieved, entry.volatility))
            print("  units     %s" % entry.units)
            print("  source    %s" % entry.citation)
        return 0

    cache_dir = args.cache_dir or receptors.DEFAULT_CACHE_DIR
    print("cache: %s\n" % cache_dir)
    failures = 0
    try:
        for entry in receptors.DATA_FILES:
            if args.verify_only:
                path = receptors.verify(entry.key, cache_dir=cache_dir)
                print("ok       %s  %s" % (entry.sha256[:16], entry.filename))
            else:
                path = receptors.download(
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
    except (receptors.ChecksumMismatch, receptors.MissingCachedFile) as error:
        print("FAILED")
        print("      %s" % error)
        failures += 1

    if failures:
        return 1
    print("")
    print("all %d file(s) verified against manifest %s."
          % (len(receptors.DATA_FILES), receptors.MANIFEST_VERSION))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
