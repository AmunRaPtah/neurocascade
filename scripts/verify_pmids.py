"""Check that every PMID quoted in a module really is the paper it claims.

    python scripts/verify_pmids.py src/neurocascade/endpoints.py

Not part of the test suite: it needs the network, and the gate runs
offline. It exists because a citation is only worth what its identifier is
worth, and the cheapest way to catch a citation that drifted from its PMID
is to ask PubMed for the title and read it back.

Prints one line per PMID: the id, the first author and year PubMed reports,
and the title. Compare against the module. Anything PubMed does not know is
printed as MISSING and is a citation to remove or fix.
"""

import json
import re
import sys
import urllib.request

ESUMMARY = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    "?db=pubmed&retmode=json&id=%s"
)


def pmids_in(path):
    with open(path) as handle:
        text = handle.read()
    return sorted(set(re.findall(r"PMID:?\s*(\d{5,9})", text)))


def main(paths):
    ids = sorted({pmid for path in paths for pmid in pmids_in(path)})
    if not ids:
        print("no PMIDs found")
        return 0
    print("%d PMIDs" % len(ids))
    for start in range(0, len(ids), 50):
        chunk = ids[start:start + 50]
        with urllib.request.urlopen(ESUMMARY % ",".join(chunk)) as response:
            payload = json.load(response)
        result = payload.get("result", {})
        for pmid in chunk:
            record = result.get(pmid)
            if not record or "error" in record:
                print("%-10s MISSING" % pmid)
                continue
            authors = record.get("authors") or [{}]
            print(
                "%-10s %-22s %-6s %s"
                % (
                    pmid,
                    (authors[0].get("name") or "?")[:22],
                    (record.get("pubdate") or "?")[:4],
                    record.get("title", "?"),
                )
            )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["src/neurocascade/endpoints.py"]))
