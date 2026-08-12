#!/usr/bin/env python
"""Regenerate the digest table pasted into
`src/neurocascade/expression/sources.py`.

This is the only place a SHA-256 for the Allen ISH expression data is
*produced*; `sources.py` is where it is *recorded*. Keeping the two apart is
the point: a digest in the manifest is a measurement of bytes that were
actually retrieved, and this script is the measuring instrument, so nobody
has to hand-type a hash and nobody is tempted to hand-edit one.

    python scripts/build_expression_manifest.py            # print the table
    python scripts/build_expression_manifest.py --check    # compare, exit 1 on drift

What it checks while it measures
--------------------------------

1. Every gene symbol in `expression/genes.py` resolves in Allen's gene table
   to the Entrez id this project recorded. A silent symbol drift (mouse
   `Drd1a` -> `Drd1` was a real one) would otherwise produce an empty map
   that looks like "this gene is not expressed".
2. Every gene's unionize query returns exactly `total_rows` rows, so the
   pinned bytes are the whole response and not the first page of a
   paginated one. The manifest records the row count for the loader to
   re-check offline.
3. Genes with zero SectionDataSets are re-queried with NO product and NO
   `failed` filter - the widest query the API allows - before being
   reported as having no data at all.

Everything it prints is either a measured number or a value copied from
`expression/genes.py`; it invents nothing.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade.expression import genes, sources  # noqa: E402

USER_AGENT = "neurocascade/P4.1"


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=600) as response:
        return response.read()


def query_json(criteria):
    url = sources._API + "query.json?" + urllib.parse.urlencode({"criteria": criteria})
    return json.loads(get(url))


def check_entrez_ids():
    """Assert every symbol resolves to the Entrez id genes.py records."""
    problems = []
    for gene in genes.RECEPTOR_GENES:
        payload = query_json(
            "model::Gene,rma::criteria,[acronym$eq'%s'],"
            "rma::options[num_rows$eq50]" % gene.symbol
        )
        entrez = set(
            row["entrez_id"]
            for row in payload["msg"]
            if row.get("organism_id") == 2 and row.get("entrez_id") is not None
        )
        if entrez != {gene.entrez_id}:
            problems.append(
                "%s: genes.py says Entrez %d, Allen mouse records say %s"
                % (gene.symbol, gene.entrez_id, sorted(entrez))
            )
        time.sleep(0.1)
    return problems


def widest_dataset_query(symbol):
    """Every SectionDataSet for a symbol: no product filter, no failed filter."""
    payload = query_json(
        "model::SectionDataSet,rma::criteria,genes[acronym$eq'%s'],"
        "rma::options[num_rows$eq200]" % symbol
    )
    return payload["total_rows"]


def measure_index():
    payload_bytes = get(sources.index_url())
    payload = json.loads(payload_bytes)
    if not payload.get("success"):
        raise SystemExit("index query failed: %s" % payload.get("msg"))
    if payload["num_rows"] != payload["total_rows"]:
        raise SystemExit(
            "index query truncated: %d of %d rows"
            % (payload["num_rows"], payload["total_rows"])
        )
    by_symbol = {}
    for row in payload["msg"]:
        for gene in row["genes"]:
            by_symbol.setdefault(gene["acronym"], []).append(
                (row["id"], row["plane_of_section"]["name"])
            )
    return (
        hashlib.sha256(payload_bytes).hexdigest(),
        len(payload_bytes),
        payload["total_rows"],
        by_symbol,
    )


def measure_gene(symbol):
    """(sha256, size, n_rows, n_datasets) for one gene's unionize CSV."""
    expected = query_json(
        sources.unionize_criteria(symbol).replace(
            "[num_rows$eq10000]", "[num_rows$eq0]"
        )
    )["total_rows"]
    payload_bytes = get(sources.unionize_url(symbol))
    lines = payload_bytes.decode("utf-8").splitlines()
    n_rows = len(lines) - 1  # header
    if n_rows != expected:
        raise SystemExit(
            "%s: CSV has %d data rows but the API reports total_rows = %d; "
            "the response is truncated and must not be pinned"
            % (symbol, n_rows, expected)
        )
    datasets = set(line.split(",")[1] for line in lines[1:])
    return (
        hashlib.sha256(payload_bytes).hexdigest(),
        len(payload_bytes),
        n_rows,
        len(datasets),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare with the committed manifest and exit 1 on any difference",
    )
    parser.add_argument(
        "--skip-entrez",
        action="store_true",
        help="skip the per-gene Entrez id cross-check (43 extra API calls)",
    )
    args = parser.parse_args(argv)

    if not args.skip_entrez:
        problems = check_entrez_ids()
        if problems:
            print("Entrez id mismatches (genes.py is wrong, or a symbol moved):")
            for problem in problems:
                print("  " + problem)
            return 1
        print("# all %d symbols resolve to the recorded Entrez ids"
              % len(genes.RECEPTOR_GENES))

    index_sha, index_size, index_rows, by_symbol = measure_index()

    empty = []
    rows = []
    for gene in genes.RECEPTOR_GENES:
        datasets = by_symbol.get(gene.symbol, [])
        if not datasets:
            total = widest_dataset_query(gene.symbol)
            empty.append((gene.symbol, total))
            continue
        rows.append((gene.symbol,) + measure_gene(gene.symbol))
        time.sleep(0.1)

    print("")
    print("_INDEX_DIGEST = (")
    print('    "%s",' % index_sha)
    print("    %d," % index_size)
    print("    %d," % index_rows)
    print(")")
    print("")
    print("_UNIONIZE_DIGESTS = (")
    for symbol, sha256, size_bytes, n_rows, n_datasets in rows:
        print(
            '    ("%s", "%s", %d, %d, %d),'
            % (symbol, sha256, size_bytes, n_rows, n_datasets)
        )
    print(")")
    print("")

    planes = {}
    for symbol, datasets in by_symbol.items():
        planes[symbol] = sorted(set(plane for _id, plane in datasets))
    sagittal_only = sorted(s for s, p in planes.items() if p == ["sagittal"])
    coronal_only = sorted(s for s, p in planes.items() if p == ["coronal"])
    print("# sagittal-only genes: %s" % (sagittal_only,))
    print("# coronal-only genes:  %s" % (coronal_only,))
    print("# genes with NO ISH data (widest query, total SectionDataSets):")
    for symbol, total in empty:
        print("#   %s: %d" % (symbol, total))

    if args.check:
        drift = []
        if (index_sha, index_size) != sources._INDEX_DIGEST[:2]:
            drift.append("index")
        committed = {row[0]: row for row in sources._UNIONIZE_DIGESTS}
        for row in rows:
            if committed.get(row[0]) != row:
                drift.append(row[0])
        for symbol in committed:
            if symbol not in set(r[0] for r in rows):
                drift.append(symbol + " (gone)")
        if drift:
            print("")
            print("MANIFEST DRIFT: %s" % ", ".join(drift))
            print(sources.VOLATILITY.strip())
            return 1
        print("")
        print("manifest %s matches upstream." % sources.MANIFEST_VERSION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
