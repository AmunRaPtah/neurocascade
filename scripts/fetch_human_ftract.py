#!/usr/bin/env python
"""Fetch and verify the F-TRACT archive plus TVB verification entries.

Downloads the F-TRACT v21-12 bundle from Zenodo and the three TVB
connectivity archives, verifying sha256 digests against the manifest
in ``neurocascade.human.ftract``.

Usage::

    python scripts/fetch_human_ftract.py
    python scripts/fetch_human_ftract.py --cache-dir /path/to/data/human
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade.human import ftract  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory (default: %s)" % ftract.DEFAULT_CACHE_DIR,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Download timeout in seconds (default: 600)",
    )
    args = parser.parse_args(argv)

    keys = [f.key for f in ftract.DATA_FILES]
    print("Fetching %d manifest entries..." % len(keys))
    for key in keys:
        print("  %s" % key)

    ftract.fetch_all(
        keys=keys,
        cache_dir=args.cache_dir,
        timeout=args.timeout,
        force=False,
        log=print,
    )
    print("Done. All verified.")


if __name__ == "__main__":
    main()
