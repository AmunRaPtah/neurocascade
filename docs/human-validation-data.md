# Human validation data workflow

The validation contracts are executable, but the repository does not invent
clinical observations. Real data must be acquired and reviewed before fitting.

## Recommended sources

- PET: OpenNeuro, NITRC, DataLad/OpenNeuro releases, and published
  supplementary tables.
- PK/BBB: FDA DailyMed, EMA EPARs, ClinicalTrials.gov records, and cited
  clinical PK/CSF studies or supplements.
- Dose-response and behavior: clinical trial supplements, regulatory review
  documents, ClinicalTrials.gov outcomes, and author-provided datasets.

Use the most specific source available and retain the DOI, registry identifier,
license, population, route, assay, and parcellation.

## Manifest discipline

`HumanDatasetManifest` records the dataset identity, layer, citation, access
status, retrieval date, checksum, and optional local path. Open and controlled
files require a SHA-256 checksum. Controlled-access or author-request data must
not be copied into the repository; commit only its manifest and derived,
non-identifying summaries. `validate_manifest_set` rejects duplicate IDs.

A manifest is not evidence by itself. Before importing observations, confirm:

- human subjects and the intended population;
- dose/exposure units and route;
- endpoint or receptor definition;
- uncertainty/error-bar semantics;
- subject duplication and study independence;
- compatible atlas/parcellation;
- redistribution permission.

## End-to-end order

1. Register candidate sources in manifests.
2. Download immutable source files without modifying them.
3. Record retrieval date and checksum.
4. Normalize into separate PK, BBB, PET, dose-response, and held-out tables.
5. Validate units, coverage, uncertainty, and provenance.
6. Reserve held-out compounds before fitting.
7. Fit only through the existing evidence-gated contracts.
8. Write aggregate results and failure reasons; never convert unavailable data
   into zeros or fitted values.

The first practical pilot should be one human drug with a PK time course, BBB
measurement, receptor occupancy or binding observation, and one supported
endpoint. Diazepam is the most complete starting candidate in the current
registry, while fentanyl and haloperidol retain explicit evidence gaps.
