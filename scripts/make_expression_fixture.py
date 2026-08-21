#!/usr/bin/env python
"""Regenerate `src/neurocascade/expression/fixtures/` from the verified cache.

Same contract as `scripts/make_connectome_fixture.py`: every fixture is a
*slice of checksum-verified published data, written back out in that file's
own format*, never a hand-written matrix of invented numbers. This script
calls `fetch.verify()` on every upstream file it reads before it reads any
of them, so a fixture can never be cut from unverified bytes.

    python scripts/fetch_expression.py          # populate data/expression/
    python scripts/make_expression_fixture.py   # rewrite the fixture directory

The slice
---------

*Genes*: eight of the 42, chosen in `FIXTURE_SYMBOLS` below for what each
one exercises rather than for coverage - see `fixtures/PROVENANCE.md`.

*Structures*: every CCF structure in the **connectome's** committed fixture
ontology (`connectome/fixtures/allen_structure_graph_1_fixture.json`, 77
structures). Reusing that ontology rather than cutting a second one is the
whole point of P4.1 sharing the connectome's structure graph: if the two
fixtures could drift apart, the tests would stop testing the alignment they
exist to test.

Rows for structures the experiment does not report are, of course, simply
absent - which is what makes the NaN path testable offline on real data
rather than on a contrived file.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade.legacy import connectome  # noqa: E402
from neurocascade.legacy.expression import fetch, sources  # noqa: E402
from neurocascade.legacy.expression import fixtures as expression_fixtures  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--out",
        default=expression_fixtures.FIXTURE_DIR,
        help="where to write (default: the committed fixture directory)",
    )
    args = parser.parse_args(argv)

    symbols = list(expression_fixtures.FIXTURE_SYMBOLS)

    # Verify EVERY upstream file before reading ANY of them.
    keys = [sources.DATASET_INDEX_KEY] + [sources.unionize_key(s) for s in symbols]
    paths = {}
    for key in keys:
        paths[key] = fetch.verify(key, cache_dir=args.cache_dir)
        print("verified %s" % os.path.basename(paths[key]))

    ontology = connectome.load_fixture_ontology()
    keep_ids = set(ontology.structures)
    print("%d structures from the connectome fixture ontology" % len(keep_ids))

    # --- the index --------------------------------------------------------
    with open(paths[sources.DATASET_INDEX_KEY], "r") as handle:
        payload = json.load(handle)
    wanted = set(symbols)
    rows = [
        row
        for row in payload["msg"]
        if any(gene["acronym"] in wanted for gene in row["genes"])
    ]
    trimmed = {
        "success": True,
        "id": payload.get("id"),
        "start_row": 0,
        "num_rows": len(rows),
        "total_rows": len(rows),
        "msg": rows,
    }
    index_out = os.path.join(args.out, expression_fixtures.INDEX_FIXTURE_NAME)
    with open(index_out, "w") as handle:
        json.dump(trimmed, handle, indent=1, sort_keys=True)
        handle.write("\n")
    print("wrote %s (%d experiments)" % (os.path.basename(index_out), len(rows)))

    # --- the unionize CSVs ------------------------------------------------
    counts = {}
    for symbol in symbols:
        source_path = paths[sources.unionize_key(symbol)]
        out_path = os.path.join(
            args.out, expression_fixtures.unionize_fixture_name(symbol)
        )
        datasets = set()
        n_rows = 0
        with open(source_path, "r", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            kept = []
            for line in reader:
                if int(line[2]) in keep_ids:
                    kept.append(line)
                    datasets.add(line[1])
                    n_rows += 1
        with open(out_path, "w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(kept)
        counts[symbol] = (n_rows, len(datasets))
        print(
            "wrote %s (%d rows, %d experiments)"
            % (os.path.basename(out_path), n_rows, len(datasets))
        )

    print("")
    print("# paste into fixtures/__init__.py:")
    print("FIXTURE_COUNTS = {")
    for symbol in symbols:
        print('    "%s": (%d, %d),' % (symbol, counts[symbol][0], counts[symbol][1]))
    print("}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
