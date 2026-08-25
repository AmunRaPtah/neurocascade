#!/usr/bin/env python
"""Regenerate the committed autoradiography fixtures from the cache.

Reads the checksum-verified Zilles & Palomero-Gallagher 2017 files out of
`data/human/` and copies the five machine-readable ones BYTE-FOR-BYTE into
`src/neurocascade/human/fixtures/` under their fixture names:

* `zilles2017_ReceptorNames_fixture.npy`
* `zilles2017_RegionNames_fixture.npy`
* `zilles2017_ReceptData_S_fixture.npy` (supragranular)
* `zilles2017_ReceptData_G_fixture.npy` (granular)
* `zilles2017_ReceptData_I_fixture.npy` (infragranular)

They can be copies rather than slices because each is one small array to
begin with, so a fixture's sha256 equals its manifest entry's. The sixth
manifest file, the Supplementary Table 2 xlsx, is deliberately NOT
shipped: fixtures must stay readable without openpyxl, and its contents
are pinned against the arrays by a cache-present test instead.

    python scripts/make_human_autoradiography_fixture.py
"""

import argparse
import hashlib
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade.human import autoradiography  # noqa: E402

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "src",
    "neurocascade",
    "human",
    "fixtures",
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the committed autoradiography fixtures from "
            "data/human."
        )
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="where the verified cache lives (default: data/human/)",
    )
    args = parser.parse_args(argv)

    cache_dir = args.cache_dir or autoradiography.DEFAULT_CACHE_DIR
    os.makedirs(FIXTURE_DIR, exist_ok=True)

    for entry in autoradiography.DATA_FILES:
        source = autoradiography.cache_path(entry.key, cache_dir)
        if not os.path.exists(source):
            print(
                "MISSING %s in %s; fetch it first "
                "(scripts/fetch_human_autoradiography.py)"
                % (entry.key, cache_dir)
            )
            return 1
        # The xlsx stays unshipped: see this script's docstring.
        if not entry.filename.endswith(".npy"):
            print("skipped (not npy)   %s" % entry.filename)
            continue
        destination = os.path.join(
            FIXTURE_DIR,
            entry.filename.replace(".npy", "_fixture.npy"),
        )
        shutil.copyfile(source, destination)
        with open(destination, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        note = (
            "byte-identical to the verified source"
            if digest == entry.sha256
            else "DIFFERS FROM MANIFEST (%s)" % digest
        )
        print("wrote %-44s %s" % (os.path.basename(destination), note))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
