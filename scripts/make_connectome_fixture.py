#!/usr/bin/env python
"""Regenerate the committed connectome test fixtures from the verified cache.

The fixtures are REAL DATA, not synthetic: each one is a slice of the
checksum-verified published file, in the published file's own format, so
the tests exercise the same parsers on the same bytes-shape that production
loads. What makes them committable is that they are fifteen regions instead
of 291, which is a few kilobytes instead of a few megabytes.

Why fifteen, and why these fifteen
----------------------------------

They are the connectome regions that this codebase's existing hand-built
scenarios already name, plus the basal-ganglia and cortical partners those
scenarios' nodes stand for:

    VTA, ACB (nucleus accumbens), PL (prelimbic, the 'PFC' node)
        the mu-opioid reward circuit
    CP (caudoputamen), GPe, STN, GPi, SNr
        the D2 indirect pathway, including the GPi node whose saturation
        is the standing lesson recorded in docs/architecture.md
    PAG, LC
        the analgesia and sedation arms of the multi-region opioid scenario
    BLA, CEA
        the GABA-A / benzodiazepine amygdala circuit
    MOp, MD, DR
        cortical and thalamic and raphe partners, so the fixture is not
        confined to one major division

That choice makes the fixture immediately useful to P3.3 (mapping the
hand-built scenarios onto connectome regions) instead of being test-only
scaffolding, and it means the numbers a test asserts on are numbers about
regions this project actually models.

LC is deliberately in the set even though Oh et al. 2014 has no LC row or
column. The Oh fixture therefore has fourteen regions where the Knox
fixture has fifteen, which is not an accident to be tidied away: it is the
smallest faithful reproduction of a real property of the two datasets, and
it keeps a test honest about the "region missing from one dataset" path.
STN earns its place twice over: it is a real node of the indirect pathway,
and it is what gives the Oh slice genuinely zero-weight connections, hence
the '#NUM!' p-value cells that exercise the reader's error-token path.

Usage:  python scripts/make_connectome_fixture.py [--cache-dir DIR]
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade.connectome import fetch, fixtures, matrix  # noqa: E402
from neurocascade.connectome import ontology as ontology_module  # noqa: E402
from neurocascade.connectome import sources  # noqa: E402

# The region set, the fixture directory and the fixture filenames all live in
# the package (`neurocascade.connectome.fixtures`), so this script and the
# tests cannot drift apart about what the fixtures are.
FIXTURE_DIR = fixtures.FIXTURE_DIR
FIXTURE_REGION_IDS = list(fixtures.FIXTURE_REGION_IDS)

KNOX_FIXTURE_KEY = fixtures.FIXTURE_CONNECTOME_KEY
OH_FIXTURE_KEY = fixtures.FIXTURE_OH_KEY

KNOX_FIXTURE_PATH = fixtures.KNOX_FIXTURE_PATH
OH_FIXTURE_PATH = fixtures.OH_FIXTURE_PATH
ONTOLOGY_FIXTURE_PATH = fixtures.ONTOLOGY_FIXTURE_PATH


def write_knox_fixture(full, path):
    """Slice the Knox CSV to the fixture regions, in the published layout."""
    small = full.subset(FIXTURE_REGION_IDS)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + [str(h) for h in small.target_hemispheres])
        writer.writerow([""] + [str(int(i)) for i in small.target_ids])
        for row, source_id in enumerate(small.source_ids):
            writer.writerow(
                # repr(float(...)) is Python's shortest round-tripping form,
                # so the fixture holds the published double exactly rather
                # than a truncated decimal that would make a value assertion
                # a test of the formatter.
                [str(int(source_id))] + [repr(float(v)) for v in small.weights[row]]
            )
    return small


def write_oh_fixture(full, path, ontology):
    """Slice Oh Supplementary Table 3 to the fixture regions it contains."""
    from openpyxl import Workbook

    available = [i for i in FIXTURE_REGION_IDS if i in set(full.source_ids.tolist())]
    small = full.subset(available)
    acronyms = [ontology[i].acronym for i in small.source_ids]
    half = small.n_target // 2

    workbook = Workbook()
    workbook.remove(workbook.active)
    blocks = [
        ("W_ipsi", small.weights[:, :half]),
        ("PValue_ipsi", small.pvalues[:, :half]),
        ("W_contra", small.weights[:, half:]),
        ("PValue_contra", small.pvalues[:, half:]),
    ]
    for name, block in blocks:
        sheet = workbook.create_sheet(name)
        sheet.append([None] + acronyms)
        for row, acronym in enumerate(acronyms):
            # A NaN p-value in memory came from a literal '#NUM!' cell in the
            # published sheet, and it has to go back as that same token.
            # Writing the float NaN instead makes openpyxl emit an EMPTY
            # cell, which the reader would have to interpret - and "blank
            # means zero" in a p-value column is the precise misreading the
            # reader refuses. So the round trip is token -> NaN -> token.
            sheet.append(
                [acronym]
                + [
                    "#NUM!" if np.isnan(v) else float(v)
                    for v in block[row]
                ]
            )
    workbook.save(path)
    return small


def write_ontology_fixture(ontology, path):
    """Trim the structure graph to the fixture regions plus their ancestors.

    Same nested shape as the published file, so `StructureOntology.from_json`
    parses fixture and production with one code path.
    """
    keep = set()
    for structure_id in FIXTURE_REGION_IDS:
        keep.update(ontology.ancestors(structure_id, include_self=True))
    # Keep every major division too, so `major_divisions()` resolves on the
    # fixture and the division-lookup test is not testing a special case.
    for division_id in ontology_module.MAJOR_DIVISION_IDS:
        keep.update(ontology.ancestors(division_id, include_self=True))

    def build(structure_id):
        structure = ontology[structure_id]
        return {
            "id": structure.id,
            "atlas_id": None,
            "ontology_id": 1,
            "acronym": structure.acronym,
            "name": structure.name,
            "color_hex_triplet": structure.color_hex_triplet,
            "graph_order": structure.graph_order,
            "st_level": structure.st_level,
            "hemisphere_id": 3,
            "parent_structure_id": structure.parent_id,
            "children": [
                build(child)
                for child in sorted(ontology.children(structure_id))
                if child in keep
            ],
        }

    roots = [s for s in keep if ontology[s].parent_id is None]
    payload = {
        "success": True,
        "id": 0,
        "start_row": 0,
        "num_rows": 1,
        "total_rows": 1,
        "msg": [build(root) for root in sorted(roots)],
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=False)
        handle.write("\n")
    return len(keep)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args(argv)

    for key in (sources.ONTOLOGY_KEY, KNOX_FIXTURE_KEY, OH_FIXTURE_KEY):
        fetch.verify(key, cache_dir=args.cache_dir)

    ontology = ontology_module.load_ontology(cache_dir=args.cache_dir)
    knox = matrix.load_connectome(
        key=KNOX_FIXTURE_KEY, ontology=ontology, cache_dir=args.cache_dir
    )
    small_knox = write_knox_fixture(knox, KNOX_FIXTURE_PATH)
    print(
        "%s: %d sources x %d targets (from %d x %d)"
        % (
            os.path.basename(KNOX_FIXTURE_PATH),
            small_knox.n_source,
            small_knox.n_target,
            knox.n_source,
            knox.n_target,
        )
    )

    oh = matrix.load_connectome(
        key=OH_FIXTURE_KEY, ontology=ontology, cache_dir=args.cache_dir
    )
    small_oh = write_oh_fixture(oh, OH_FIXTURE_PATH, ontology)
    print(
        "%s: %d sources x %d targets (from %d x %d)"
        % (
            os.path.basename(OH_FIXTURE_PATH),
            small_oh.n_source,
            small_oh.n_target,
            oh.n_source,
            oh.n_target,
        )
    )

    kept = write_ontology_fixture(ontology, ONTOLOGY_FIXTURE_PATH)
    print(
        "%s: %d of %d structures"
        % (os.path.basename(ONTOLOGY_FIXTURE_PATH), kept, len(ontology))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
