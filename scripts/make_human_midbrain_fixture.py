#!/usr/bin/env python
"""Regenerate the committed CIT168 midbrain-atlas fixture from the cache.

Reads the checksum-verified CIT168 release bundle out of `data/human/`,
extracts the label table member, re-verifies THAT against the inner digest
recorded in `midbrain.CIT168_TAR_MEMBERS`, and copies it BYTE-FOR-BYTE to:

    src/neurocascade/human/fixtures/cit168_pauli2018_info_fixture.csv

The bundle's image members are deliberately NOT shipped: they are two
orders of magnitude larger than every other committed fixture combined,
nothing offline needs them, and -- since the deposited images integrate
systematically differently from the paper's own Table 1 (see
`midbrain.DEPOSIT_DISCREPANCY_NOTE`) -- no product value derives from
them anyway. When the cache is present, the cache-present tests parse the
probabilistic member and reproduce the recorded discrepancy numbers.

    python scripts/make_human_midbrain_fixture.py
"""

import argparse
import hashlib
import os
import sys
import tarfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade.human import midbrain  # noqa: E402

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
            "Regenerate the committed CIT168 info.csv fixture from "
            "data/human."
        )
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="where the verified cache lives (default: data/human/)",
    )
    args = parser.parse_args(argv)

    cache_dir = args.cache_dir or midbrain.DEFAULT_CACHE_DIR
    bundle = midbrain.cache_path("cit168_pauli2018_atlas_bundle", cache_dir)
    if not os.path.exists(bundle):
        print(
            "MISSING %s; fetch it first (scripts/fetch_human_midbrain.py)"
            % bundle
        )
        return 1
    verified = midbrain.verify(
        "cit168_pauli2018_atlas_bundle", cache_dir=cache_dir
    )
    assert verified == bundle

    expected_sha, expected_size = midbrain.CIT168_TAR_MEMBERS[
        midbrain.CIT168_INFO_MEMBER
    ]
    with tarfile.open(verified, "r:gz") as archive:
        data = archive.extractfile(midbrain.CIT168_INFO_MEMBER).read()
    actual_sha = hashlib.sha256(data).hexdigest()
    if (actual_sha, len(data)) != (expected_sha, expected_size):
        print(
            "REFUSING: member %s of the verified bundle has sha256 %s "
            "(%d bytes), expected %s (%d bytes)"
            % (
                midbrain.CIT168_INFO_MEMBER,
                actual_sha, len(data), expected_sha, expected_size,
            )
        )
        return 1

    destination = os.path.join(FIXTURE_DIR, "cit168_pauli2018_info_fixture.csv")
    with open(destination, "wb") as handle:
        handle.write(data)
    print(
        "wrote %s (%d bytes) byte-identical to the verified member"
        % (os.path.basename(destination), len(data))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
