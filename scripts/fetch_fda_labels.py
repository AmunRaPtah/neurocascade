#!/usr/bin/env python3
"""Snapshot FDA approved drug labels from openFDA, for the label-dose test.

Why the FDA and not the literature
----------------------------------

P22.2 needs the approved daily dose range for schizophrenia per drug, to test
the chain's predicted 65-to-78-percent D2 occupancy window against a number
nobody in this repository chose. The approved range is genuine held-out ground
truth: it was set by the registration trials and has never entered any fit here.
It is also, unlike most of what this project needs, free and unpaywalled. Every
searched-and-empty row in the validation ledger exists because a primary paper
was behind a paywall.

openFDA (api.fda.gov/drug/label.json) serves the current Structured Product
Label. It needs no API key at this volume. The Drugs@FDA approval packages,
which carry the Clinical Pharmacology and Biopharmaceutics Reviews that P21.3
needs for population PK, are a separate surface and are not fetched here: they
are PDFs behind a per-application browse UI, so they are retrieved by hand and
recorded with their application number and review date.

What this script deliberately does not do
-----------------------------------------

It does not parse a dose out of the label. The DOSAGE AND ADMINISTRATION section
is prose, it distinguishes initial from maintenance from maximum, it separates
acute from chronic and adult from adolescent, and for several of these drugs it
gives a different range per indication. A regular expression over that text
would produce a number that looks sourced and is not, which is the single
failure mode this project has spent the most effort avoiding.

So the snapshot is the deliverable. The script writes the verbatim label bytes
plus their SHA-256, and the transcription into a dose range is a separate,
attributed step that records the sentence it came from.

Usage
-----

    scripts/fetch_fda_labels.py                    # the drugs in DRUGS
    scripts/fetch_fda_labels.py haloperidol clozapine
    scripts/fetch_fda_labels.py --show haloperidol # print the dosage section
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "data" / "fda"
ENDPOINT = "https://api.fda.gov/drug/label.json"
USER_AGENT = "neurocascade/fda (research; contact via repository)"

#: Antipsychotics with published human D2 PET occupancy data, which is the
#: constraint on the drug set rather than anything about the label.
DRUGS = [
    "haloperidol",
    "risperidone",
    "paliperidone",
    "olanzapine",
    "quetiapine",
    "ziprasidone",
    "aripiprazole",
    "clozapine",
    "amisulpride",
]


def _get(params: dict) -> dict:
    url = ENDPOINT + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode())


def fetch_labels(generic: str, limit: int = 100) -> list[dict]:
    """Every current label openFDA holds for this generic name.

    Returns [] rather than raising on a 404, which is how openFDA reports "no
    matches" and is a real answer, not an error.
    """
    try:
        return _get({
            "search": f'openfda.generic_name:"{generic}"',
            "limit": str(limit),
        })["results"]
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return []
        raise


#: The chain predicts an oral daily dose, so an injectable label answers a
#: different question. This filter is not a detail: the first version of this
#: script preferred the most recently revised NDA, and that silently returned
#: Haldol Decanoate, Zyprexa Relprevv and Erzofri, all long-acting
#: intramuscular, because recent antipsychotic development has gone into depot
#: formulations. Three of nine drugs would have been tested against a depot
#: dosing interval as though it were a daily oral dose.
REQUIRED_ROUTE = "ORAL"

#: And the label must be for the indication the occupancy threshold came from.
#: Kapur 2000's 65-percent response threshold is a schizophrenia finding.
#:
#: Two accepted phrasings, not one, and the reason is a fact about the labels
#: rather than a loosened criterion. Requiring the literal word "schizophrenia"
#: rejected haloperidol, which is the drug the whole chain is validated on. All
#: 68 of its current US oral labels use the older FDA wording, "management of
#: manifestations of psychotic disorders", because haloperidol was approved in
#: 1967 and its label was never rewritten to the modern vocabulary. Which
#: phrase matched is recorded per drug, so the older wording is visible rather
#: than absorbed.
#:
#: This filter is what catches amisulpride, whose only US application is
#: Barhemsys, intravenous, for postoperative nausea and vomiting: it is not
#: approved as an antipsychotic in the United States at all, under either
#: phrasing.
ACCEPTED_INDICATIONS = ("schizophrenia", "psychotic disorder")


def choose_label(labels: list[dict]) -> tuple[dict | None, str]:
    """Pick the oral, schizophrenia-indicated innovator label, and say why.

    An NDA is the original application, whose dose range was set by the
    registration trials the FDA reviewed. An ANDA is a generic whose label is
    copied from the innovator's and may be a stale copy. Prefer the NDA, but
    only after route and indication, because a generic oral tablet answers the
    question and an innovator depot injection does not.
    """
    def applications(label: dict) -> list[str]:
        return label.get("openfda", {}).get("application_number", []) or []

    def routes(label: dict) -> list[str]:
        return label.get("openfda", {}).get("route", []) or []

    def indication_match(label: dict) -> str | None:
        text = " ".join(label.get("indications_and_usage", []) or []).lower()
        return next((p for p in ACCEPTED_INDICATIONS if p in text), None)

    with_dosage = [x for x in labels if x.get("dosage_and_administration")]
    if not with_dosage:
        return None, "no label carries a DOSAGE AND ADMINISTRATION section"

    oral = [x for x in with_dosage if REQUIRED_ROUTE in routes(x)]
    if not oral:
        seen = sorted({r for x in with_dosage for r in routes(x)})
        return None, (
            f"no oral label: {len(with_dosage)} label(s) with dosing, "
            f"route(s) {seen or ['unrecorded']}. The chain predicts an oral "
            "daily dose, so an injectable label answers a different question.")

    on_indication = [x for x in oral if indication_match(x)]
    if not on_indication:
        return None, (
            f"{len(oral)} oral label(s), none indicated for any of "
            f"{list(ACCEPTED_INDICATIONS)}. The 65-percent occupancy threshold "
            "is a schizophrenia finding, so another indication's dose range is "
            "not the ground truth this test needs.")

    ndas = [x for x in on_indication
            if any(a.startswith("NDA") for a in applications(x))]
    pool, kind = (ndas, "NDA") if ndas else (on_indication, "ANDA only")

    # Most recently revised within the preferred kind: labels are amended, and
    # the current dose range is the one on the current revision.
    chosen = max(pool, key=lambda x: x.get("effective_time", ""))
    reason = (
        f"{kind}, oral, indicated as '{indication_match(chosen)}'; of "
        f"{len(labels)} matched label(s), {len(oral)} oral, "
        f"{len(on_indication)} on indication, {len(ndas)} of those NDA; "
        "chose the most recently revised"
    )
    if not ndas:
        reason += (
            ". No NDA oral label is current for this drug, so this is a "
            "generic's copy of a withdrawn innovator label")
    return chosen, reason


def snapshot(generic: str) -> dict:
    """Write one drug's label to the cache and return its provenance record."""
    CACHE.mkdir(parents=True, exist_ok=True)
    labels = fetch_labels(generic)
    chosen, reason = choose_label(labels)

    if chosen is None:
        # Searched and empty is a result. Record it in the same shape.
        return {
            "generic": generic, "status": "searched-and-empty",
            "reason": reason or "openFDA returned no label for this name",
            "endpoint": ENDPOINT, "matched": len(labels),
        }

    payload = json.dumps(chosen, indent=2, sort_keys=True).encode()
    path = CACHE / f"label-{generic}.json"
    path.write_bytes(payload)

    dosage = "\n\n".join(chosen.get("dosage_and_administration", []))
    (CACHE / f"dosage-{generic}.txt").write_text(dosage)

    openfda = chosen.get("openfda", {})
    return {
        "generic": generic, "status": "ok",
        "application_number": openfda.get("application_number", []),
        "brand_name": openfda.get("brand_name", []),
        "effective_time": chosen.get("effective_time"),
        "spl_set_id": chosen.get("set_id"),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "dosage_chars": len(dosage),
        "selection": reason,
        "endpoint": ENDPOINT,
        "path": str(path.relative_to(REPO)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("drugs", nargs="*", default=None)
    parser.add_argument("--show", metavar="DRUG",
                        help="print a cached dosage section and exit")
    args = parser.parse_args()

    if args.show:
        path = CACHE / f"dosage-{args.show}.txt"
        if not path.exists():
            print(f"not cached: {path}. Run without --show first.", file=sys.stderr)
            return 1
        print(path.read_text())
        return 0

    records = []
    for generic in (args.drugs or DRUGS):
        try:
            record = snapshot(generic)
        except Exception as exc:  # noqa: BLE001 - one drug must not stop the rest
            record = {"generic": generic, "status": "error", "reason": repr(exc)}
        records.append(record)
        mark = {"ok": "  ", "searched-and-empty": "--", "error": "!!"}[record["status"]]
        detail = (", ".join(record.get("application_number") or []) or record.get("reason", ""))
        print(f"{mark} {generic:<14} {record.get('effective_time', ''):<10} {detail}")
        time.sleep(0.5)  # openFDA rate-limits unkeyed clients

    index = CACHE / "index.json"
    index.write_text(json.dumps(records, indent=2) + "\n")
    ok = sum(1 for r in records if r["status"] == "ok")
    print(f"\n{ok}/{len(records)} snapshotted; provenance in {index.relative_to(REPO)}")
    print("The dose range is NOT parsed from these. Transcribe it by hand, "
          "recording the sentence it came from.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
