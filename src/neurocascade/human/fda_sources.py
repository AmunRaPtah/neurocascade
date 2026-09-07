"""Pinned provenance for the FDA label snapshots behind the P22.2 dose test.

`data/` is gitignored in full, exactly as for the connectome manifest in
`sources.py`, so this module is the committed record: which application number
each dose range came from, when that label was last revised, and the SHA-256 of
the bytes that were read. `scripts/fetch_fda_labels.py` writes the snapshots;
this file is what makes them checkable afterwards.

Two entries are `searched-and-empty`, and both are real findings rather than
gaps to be filled later.

Amisulpride has no US antipsychotic approval at all. Its only application is
Barhemsys, intravenous, for postoperative nausea and vomiting. It is widely used
as an antipsychotic in Europe and has good D2 PET data, so it stays in the drug
set for the PET test in P22.3 and is excluded only from the label-dose test in
P22.2, where the US route cannot supply ground truth.

Haloperidol has no current innovator oral label: Haldol oral tablets were
discontinued, and all 68 current US oral labels are ANDAs, generics carrying a
copy of a withdrawn label. Those labels also predate the modern indication
wording and say "management of manifestations of psychotic disorders" rather
than "schizophrenia". Both facts are recorded per record rather than smoothed
over, because haloperidol is the one drug the chain is validated on.

Labels are revised, so `effective_time` and `sha256` will drift. That is the
point of pinning them: a changed hash means the dose range must be re-read, not
that the fetch is broken.
"""

from __future__ import annotations

#: Date every snapshot below was retrieved, in one session.
RETRIEVED = "2026-09-05"

#: openFDA label endpoint. No API key is needed at this volume.
ENDPOINT = "https://api.fda.gov/drug/label.json"

#: The approval packages carrying the Clinical Pharmacology and Biopharmaceutics
#: Reviews that P21.3 needs for population PK.
#:
#: This comment used to say the packages were "not machine-fetchable: they are
#: PDFs behind a per-application browse UI". P21.3 fetched 26 of them
#: non-interactively and the claim is withdrawn. What was actually blocking the
#: fetch, in order of how long each cost: the per-year document index is
#: ``drugsatfda_docs/nda/<year>/<...>toc.html`` and not the ``.cfm`` the browse
#: UI links to; sending a browser User-Agent trips the FDA WAF, which answers
#: 302 to ``abuse-detection-apology.html`` while curl's default agent is served
#: normally; the review filenames (ClinPharmR, BioPharmR, MultidisciplineR,
#: IntegratedR) vary in case across years; and post-2020 tables of contents
#: carry no literal links at all, synthesising the filenames in JavaScript from
#: a ``pdfBaseName`` variable.
#:
#: The barrier that remains is inside the PDFs, not in front of them: sponsor
#: final-parameter tables are usually raster images, which pdftotext skips
#: without any signal that it did. See docs/p21-3-fda-population-pk.md.
DRUGS_AT_FDA = "https://www.accessdata.fda.gov/scripts/cder/daf/"

LABELS = [
    dict(
        generic='haloperidol',
        status="ok",
        application_number=['ANDA216918'],
        brand_name=['Haloperidol'],
        effective_time='20260831',
        spl_set_id='a0751f2e-05ab-4ba9-9bae-7b791bf1a621',
        sha256='efecfcc21ff356fbc3d38a50a3cac75a86d9b9f74b72a3fcd728ab67bea3f32f',
        bytes=43053,
        selection=(
            "ANDA only, oral, indicated as 'psychotic disorder'; of 100 "
            'matched label(s), 68 oral, 68 on indication, 0 of those NDA; '
            'chose the most recently revised. No NDA oral label is '
            "current for this drug, so this is a generic's copy of a "
            'withdrawn innovator label'
        ),
    ),
    dict(
        generic='risperidone',
        status="ok",
        application_number=['NDA020588'],
        brand_name=['Risperidone'],
        effective_time='20260820',
        spl_set_id='701e1cd8-2868-4d29-85a9-a2212c88165f',
        sha256='ddeddf068a6e2c80e4527f48b63fec7e4458340bdb463b523adaddc49c8a2187',
        bytes=220648,
        selection=(
            "NDA, oral, indicated as 'schizophrenia'; of 75 matched "
            'label(s), 68 oral, 68 on indication, 3 of those NDA; chose '
            'the most recently revised'
        ),
    ),
    dict(
        generic='paliperidone',
        status="ok",
        application_number=['NDA021999'],
        brand_name=['INVEGA'],
        effective_time='20250131',
        spl_set_id='7b8e5b26-b9e4-4704-921b-3c3c0d159916',
        sha256='6654ef465d54fd4c49b1a9b2d658144c62b13722aafd10dd80d1eda783f39b2c',
        bytes=166338,
        selection=(
            "NDA, oral, indicated as 'schizophrenia'; of 29 matched "
            'label(s), 25 oral, 25 on indication, 1 of those NDA; chose '
            'the most recently revised'
        ),
    ),
    dict(
        generic='olanzapine',
        status="ok",
        application_number=['NDA020592', 'NDA021253', 'NDA021086'],
        brand_name=['Zyprexa', 'ZYPREXA Intramuscular', 'ZYPREXA Zydis'],
        effective_time='20260211',
        spl_set_id='b418946a-1ab4-4d89-a012-d94fc361a3c4',
        sha256='397beb24e6aaa67921e4b1baab8ef08533af081d86728498cc60dd7cc273d36b',
        bytes=326502,
        selection=(
            "NDA, oral, indicated as 'schizophrenia'; of 100 matched "
            'label(s), 87 oral, 85 on indication, 2 of those NDA; chose '
            'the most recently revised'
        ),
    ),
    dict(
        generic='quetiapine',
        status="ok",
        application_number=['NDA020639'],
        brand_name=['SEROQUEL'],
        effective_time='20260410',
        spl_set_id='8ae62aaf-f8c5-417f-baac-0098369ca322',
        sha256='36b5e2c0a98349c7040367cfb5b1367f8d5dbb1cbaa9ce6f07896387f73b8f67',
        bytes=356195,
        selection=(
            "NDA, oral, indicated as 'schizophrenia'; of 100 matched "
            'label(s), 100 oral, 100 on indication, 3 of those NDA; chose '
            'the most recently revised'
        ),
    ),
    dict(
        generic='ziprasidone',
        status="ok",
        application_number=['NDA020825', 'NDA020919'],
        brand_name=['Geodon'],
        effective_time='20250115',
        spl_set_id='a2b4cc65-6a6d-49c4-adfc-a4458562a167',
        sha256='ba793915f2fb74ba29dfb37bb37679cfe295fbdbf4efde773a2999c7840417b9',
        bytes=237377,
        selection=(
            "NDA, oral, indicated as 'schizophrenia'; of 41 matched "
            'label(s), 33 oral, 33 on indication, 2 of those NDA; chose '
            'the most recently revised'
        ),
    ),
    dict(
        generic='aripiprazole',
        status="ok",
        application_number=['NDA216655'],
        brand_name=['OPIPZA'],
        effective_time='20250130',
        spl_set_id='b47ede68-8942-4432-af7e-57afc14b2ba4',
        sha256='d9fc70b6c354a8cd4a93fd3ae74daf8d171c29955539cf74389004b64cc60425',
        bytes=231544,
        selection=(
            "NDA, oral, indicated as 'schizophrenia'; of 100 matched "
            'label(s), 97 oral, 97 on indication, 2 of those NDA; chose '
            'the most recently revised'
        ),
    ),
    dict(
        generic='clozapine',
        status="ok",
        application_number=['NDA203479'],
        brand_name=['VERSACLOZ'],
        effective_time='20250623',
        spl_set_id='2592c9a8-fd74-4e0d-a895-b07b014cf355',
        sha256='167d7d970c5450a4a6defb0eb0e4c7159c2d40eba53cf84d3bdda9fe659da419',
        bytes=154541,
        selection=(
            "NDA, oral, indicated as 'schizophrenia'; of 16 matched "
            'label(s), 16 oral, 16 on indication, 2 of those NDA; chose '
            'the most recently revised'
        ),
    ),
    dict(
        generic='amisulpride',
        status="searched-and-empty",
        reason=(
            'no oral label: 1 label(s) with dosing, route(s) '
            "['INTRAVENOUS']. The chain predicts an oral daily dose, so "
            'an injectable label answers a different question.'
        ),
    ),
]

LABELS_BY_DRUG = {r['generic']: r for r in LABELS}

#: Drugs with a usable US oral label. This is the drug set P22.2 can test,
#: and it is smaller than the set P22.3 can test against PET.
TESTABLE = [r['generic'] for r in LABELS if r['status'] == 'ok']
