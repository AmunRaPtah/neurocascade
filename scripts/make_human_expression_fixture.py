#!/usr/bin/env python
"""Regenerate the committed human AHBA expression fixture from the cache.

Reads the checksum-verified full AHBA matrix out of `data/human/` and writes
the twelve-region slice (the same twelve regions as the connectome fixture)
times the forty stable receptor genes, under
`src/neurocascade/human/fixtures/`. The region and gene lists are defined
once, here and in `expression.py`, so the fixture files and
`fixtures/PROVENANCE.md` cannot disagree with the loader.

    python scripts/make_human_expression_fixture.py
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade.human import expression  # noqa: E402

#: The twelve fixture regions, in published file order. Identical to the
#: connectome fixture's regions; see `fixtures/PROVENANCE.md`.
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

OUT_FILENAME = "enigma_hcp_ahba_dk_expression_fixture.csv"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Regenerate the human AHBA expression fixture from the cache."
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    path = expression.ensure(
        expression.DEFAULT_EXPRESSION_KEY,
        cache_dir=args.cache_dir,
        allow_download=True,
    )

    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)

    gene_columns = header[1:]
    region_labels = [row[0] for row in rows]

    missing_regions = [label for label in FIXTURE_LABELS if label not in region_labels]
    if missing_regions:
        raise SystemExit("fixture regions absent from the file: %s" % missing_regions)
    missing_genes = [symbol for symbol in expression.STABLE_SYMBOLS if symbol not in gene_columns]
    if missing_genes:
        raise SystemExit("stable genes absent from the file: %s" % missing_genes)

    row_index = {label: i for i, label in enumerate(region_labels)}
    col_index = {symbol: gene_columns.index(symbol) for symbol in expression.STABLE_SYMBOLS}

    out_path = os.path.join(args.out_dir, OUT_FILENAME)
    with open(out_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label"] + list(expression.STABLE_SYMBOLS))
        for label in FIXTURE_LABELS:
            row = rows[row_index[label]]
            writer.writerow(
                [label] + [row[col_index[symbol] + 1] for symbol in expression.STABLE_SYMBOLS]
            )

    print("wrote %s" % out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
