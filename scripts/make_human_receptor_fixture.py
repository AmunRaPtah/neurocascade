#!/usr/bin/env python
"""Regenerate the committed human PET receptor fixtures from the cache.

Reads the checksum-verified Hansen 2022 files out of `data/human/` and
writes the offline test set under `src/neurocascade/human/fixtures/`:

* five tracer maps, copied BYTE-FOR-BYTE from the verified cache. They are
  one row of 83 numbers each, so "the fixture" and "the file" can be the
  same bytes, and a fixture's sha256 equals the manifest's;
* one 68 x 3 slice of the paper's deposited aggregation matrix -- the three
  ingested receptors' columns, copied as VERBATIM TEXT in
  `receptors.PUBLISHED_FIXTURE_COLUMNS` order.

The column list is defined once, here and in `receptors.py`, so the fixture
on disk and the reader that checks it cannot disagree. See PROVENANCE.md
for the source digests this depends on.

    python scripts/make_human_receptor_fixture.py
"""

import argparse
import hashlib
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade.human import receptors  # noqa: E402

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
        description="Regenerate the committed receptor fixtures from data/human."
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="where the verified cache lives (default: data/human/)",
    )
    args = parser.parse_args(argv)

    cache_dir = args.cache_dir or receptors.DEFAULT_CACHE_DIR
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    written = []

    for entry in receptors.DATA_FILES:
        source = receptors.cache_path(entry.key, cache_dir)
        if not os.path.exists(source):
            print(
                "MISSING %s in %s; fetch it first "
                "(scripts/fetch_human_receptors.py)" % (entry.key, cache_dir)
            )
            return 1
        # The xlsx stays unshipped: fixtures must stay readable without
        # openpyxl, and its content is pinned against the hard-coded PARCELS
        # table by a cache-present test instead.
        if not entry.filename.endswith(".csv"):
            print("skipped (not csv)  %s" % entry.filename)
            continue

        destination = os.path.join(
            FIXTURE_DIR, entry.filename.replace(".csv", "_fixture.csv")
        )
        if entry.key == receptors.PUBLISHED_MATRIX_KEY:
            note = _write_matrix_slice(source, destination)
        else:
            # One row of 83 numbers: the verbatim copy IS the slice.
            shutil.copyfile(source, destination)
            note = _byte_note(destination, entry)
        written.append((os.path.basename(destination), note))

    print("fixtures under %s:" % os.path.abspath(FIXTURE_DIR))
    for name, note in written:
        print("  %-64s %s" % (name, note))
    return 0


def _write_matrix_slice(source, destination):
    """Copy three whole columns as their original text."""
    columns = [
        receptors._PUBLISHED_COLUMNS[key]
        for key in receptors.PUBLISHED_FIXTURE_COLUMNS
    ]
    with open(source) as handle:
        rows = [line.split(",") for line in handle.read().strip().splitlines()]
    expected_rows = len(receptors._PUBLISHED_ROWS)
    if len(rows) != expected_rows or any(len(r) < max(columns) + 1 for r in rows):
        raise SystemExit(
            "%s: unexpected shape; regenerate after re-verifying the manifest"
            % source
        )
    with open(destination, "w") as handle:
        for row in rows:
            handle.write(",".join(row[c] for c in columns) + "\n")
    return (
        "%d x %d text-verbatim slice, columns %s"
        % (
            len(rows),
            len(columns),
            ", ".join(receptors.PUBLISHED_FIXTURE_COLUMNS),
        )
    )


def _byte_note(path, entry):
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    if digest == entry.sha256:
        return "byte-identical to the verified source"
    return "DIFFERS FROM MANIFEST (%s)" % digest


if __name__ == "__main__":
    raise SystemExit(main())
