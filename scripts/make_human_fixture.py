#!/usr/bin/env python
"""Regenerate the committed human-connectome fixture from the verified cache.

Reads the checksum-verified full files out of `data/human/` and writes the
twelve-region slice under `src/neurocascade/human/fixtures/`. The slice is
defined once, here, so the fixture files and `fixtures/PROVENANCE.md` cannot
disagree with each other.

    python scripts/make_human_fixture.py
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade.human import fetch, sources  # noqa: E402

#: The twelve fixture regions, in published file order. See
#: `src/neurocascade/human/fixtures/PROVENANCE.md`.
FIXTURE_LABELS = (
    "L_bankssts",
    "R_bankssts",
    "L_precentral",
    "R_precentral",
    "L_superiorfrontal",
    "Laccumb",
    "Lamyg",
    "Lcaud",
    "Lhippo",
    "Lpal",
    "Lput",
    "Lthal",
)

DEFAULT_OUT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "neurocascade",
        "human",
        "fixtures",
    )
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Regenerate the human-connectome fixture from the cache."
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    matrix_path = fetch.ensure(
        sources.DEFAULT_MATRIX_KEY, cache_dir=args.cache_dir, allow_download=True
    )
    labels_path = fetch.ensure(
        sources.LABELS_KEY, cache_dir=args.cache_dir, allow_download=True
    )

    with open(labels_path, newline="") as handle:
        labels = next(csv.reader(handle))
    with open(matrix_path, newline="") as handle:
        rows = list(csv.reader(handle))

    missing = [label for label in FIXTURE_LABELS if label not in labels]
    if missing:
        raise SystemExit("fixture labels absent from the published file: %s" % missing)

    indices = [labels.index(label) for label in FIXTURE_LABELS]

    label_path = os.path.join(
        args.out_dir, "enigma_hcp_sc_labels_aparc_with_sctx_fixture.csv"
    )
    matrix_out = os.path.join(
        args.out_dir, "enigma_hcp_sc_matrix_aparc_with_sctx_fixture.csv"
    )

    with open(label_path, "w", newline="") as handle:
        handle.write(",".join(labels[i] for i in indices) + "\n")
    with open(matrix_out, "w", newline="") as handle:
        for r in indices:
            handle.write(",".join(rows[r][c] for c in indices) + "\n")

    print("wrote %s" % label_path)
    print("wrote %s" % matrix_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
