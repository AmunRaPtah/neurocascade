#!/usr/bin/env python
"""Download and verify the human autoradiography data this project ingests.

Everything it knows about what to download lives in
`src/neurocascade/human/autoradiography.py`: the URLs (pinned to one
commit of AlGoulas/receptor_principles), the SHA-256 digests, the sizes,
the citations and licences, and what the numbers mean. This script is only
the command line around it, so there is exactly one place a checksum is
written down.

    python scripts/fetch_human_autoradiography.py              # fetch + verify all 6
    python scripts/fetch_human_autoradiography.py --verify-only
    python scripts/fetch_human_autoradiography.py --list       # manifest
    python scripts/fetch_human_autoradiography.py --checksums  # sha256sum -c input

Files land in `data/human/` next to the other human layers (about 76 KB:
five small .npy arrays plus the paper's Supplementary Table 2 xlsx). The
committed half of the contract is the manifest, so a clean checkout plus
this script reproduces the exact bytes or fails saying what moved.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade.human import autoradiography  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Fetch and verify the Zilles & Palomero-Gallagher 2017 "
            "autoradiography absolute-density table."
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

    if args.list:
        for entry in autoradiography.DATA_FILES:
            print("%-44s %8d bytes  %s..." % (
                entry.key, entry.size_bytes, entry.sha256[:16],
            ))
            print("    %s" % entry.url)
        return 0
    if args.checksums:
        print("\n".join(autoradiography.checksum_lines()))
        return 0

    keys = [entry.key for entry in autoradiography.DATA_FILES]
    if args.verify_only:
        for key in keys:
            path = autoradiography.verify(key, cache_dir=args.cache_dir)
            print("verified  %s -> %s" % (key, path))
        return 0

    paths = autoradiography.fetch_all(
        keys=keys, cache_dir=args.cache_dir, timeout=args.timeout,
        force=args.force,
    )
    for key, path in sorted(paths.items()):
        print("fetched   %-44s %s" % (key, path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
