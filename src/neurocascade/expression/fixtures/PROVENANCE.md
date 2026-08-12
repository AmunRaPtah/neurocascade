# Fixture provenance

Every file in this directory is a **slice of real, checksum-verified Allen
Mouse Brain Atlas ISH data**, written back out in the source response's own
format. None of it is synthetic. Regenerate with:

    python scripts/fetch_expression.py            # populate data/expression/
    python scripts/make_expression_fixture.py     # rewrite this directory

`make_expression_fixture.py` calls `fetch.verify()` on all nine upstream
files before it reads any of them, so a fixture can never be cut from
unverified bytes.

## The structure set is the connectome's, on purpose

The structures kept are exactly the **69 CCF structures in
`connectome/fixtures/allen_structure_graph_1_fixture.json`**: this package
has no ontology fixture of its own and must not grow one. P4.1's claim is
that expression and connectivity are indexed by the same CCF structure graph
1; two fixture ontologies free to drift apart would let the offline tests
keep passing after that stopped being true.

Fifteen of those 69 are the connectome fixture's regions (MOp, PL, ACB, CP,
BLA, CEA, GPe, GPi, STN, MD, VTA, SNr, PAG, DR, LC); the rest are their
ancestors, the twelve major divisions, the structures P3.2's sign table
names, P4.2's five autoradiography-comparison regions (OT, CLA, EPd, ACA,
HIP) and P4.3's seven anatomy-claim regions (CA1, CA3, DG, MH, PVT, SNc,
CBX). The set grew from 55 to 61 with P4.2 and to 69 with P4.3; each growth
recut every file in this directory from the same verified bytes.

## The eight genes and what each one is here for

| gene | experiments | why it is in the fixture |
|---|---|---|
| `Oprm1`   | 3 (1 coronal, 2 sagittal) | the receptor the model already simulates; its sagittal experiment 69860840 does not report DR, so the NaN path runs on real data |
| `Drd1`    | 4 (1 coronal, 3 sagittal) | experiment 354 reports only 37 of the 69 structures, the widest real coverage gap in the fixture |
| `Drd2`    | 3 of 4 indexed | the other modelled receptor; striatal. Its indexed experiment 359 has no unionize rows at all |
| `Gabra1`  | 2 (1 coronal, 1 sagittal) | the third modelled system; cortex-dominant. Its sagittal experiment does not report DR |
| `Htr2a`   | 1 (sagittal) | the only **sagittal-only** gene here, so `plane="coronal"` yields an all-NaN row that has to be reported, not read as "not expressed" |
| `Cnr1`    | 4 of 5 indexed | its indexed experiment 282 has no unionize rows in any structure graph |
| `Adora2a` | 3 (1 coronal, 2 sagittal) | striatal, and the most striatum-restricted receptor in P4.3's claim set |
| `Slc6a3`  | 4 (2 coronal, 2 sagittal) | the dopamine transporter, confined to VTA/SNc: the sharpest available "is this map anatomically real" check. Its sagittal experiment 79761025 does not report BLA |

## The nine files

### `aba_ish_section_data_sets_fixture.json`

26 of the 106 experiments in the pinned index: every experiment of the eight
genes above, cut from

    aba_ish_section_data_sets.json
    sha256 cf26412b7c574eebf93c8b6391f8ccbdd6bc3b050b1acfaa914bf3f268d6d43c
    106 experiments, 43 genes

Same `{"success", "num_rows", "total_rows", "msg": [...]}` envelope the API
returns, with `num_rows` and `total_rows` rewritten to the fixture's 26 so
the reader's truncation check (`num_rows == total_rows`) is exercised
honestly rather than bypassed. Each experiment record keeps every field the
API returned. Re-serialised with `json.dump(..., indent=1, sort_keys=True)`,
so the whitespace and key order are this repository's and the *values* are
Allen's (a JSON formatting difference, not a data difference).

### The eight `aba_ish_unionize_<Gene>_fixture.csv`

Each is the rows of the corresponding pinned response whose `structure_id`
is one of the 69, with the published eleven-column header unchanged:

| fixture | rows | experiments | sha256 of the pinned response |
|---|---|---|---|
| `aba_ish_unionize_Oprm1_fixture.csv`   | 206 | 3 | `a001196e12ebf5eb8f5c9f08ac84abebd4c7d640d065c84fd749f373b7f0556e` |
| `aba_ish_unionize_Drd1_fixture.csv`    | 244 | 4 | `d5a199f31aee728e18e5399d64ac0c338ff1f597ccd220b8b5473396ed0fef25` |
| `aba_ish_unionize_Drd2_fixture.csv`    | 207 | 3 | `035841983dfdff5b932bce6299acd8bb14d8ed119fea4ef09e7af08a0f0351c5` |
| `aba_ish_unionize_Gabra1_fixture.csv`  | 137 | 2 | `c3fb6bc96749c64788ff0eeb126d4792fe704bb25f6d6836fe141683470bac17` |
| `aba_ish_unionize_Htr2a_fixture.csv`   |  69 | 1 | `0e54f8ae2912e7d72bc11e6966d67c58e55e25e34e58f64355d2ac4d10b82717` |
| `aba_ish_unionize_Cnr1_fixture.csv`    | 276 | 4 | `abdeeba67ea6bd6d360f02a7e1af9c30feef3036c122781bb9776e8ad2cbffdd` |
| `aba_ish_unionize_Adora2a_fixture.csv` | 207 | 3 | `6af587c8701268060eb8d52ed0f65f9a1f440cd7afdb03429952752327e4e8d2` |
| `aba_ish_unionize_Slc6a3_fixture.csv`  | 275 | 4 | `5183aa633c14a0d9f0030679c32e08516ecc65b5a437ffb0f282d83c7d58b41d` |

**Character-identical to the pinned bytes.** The rows are copied through
`csv.reader`/`csv.writer` without ever being parsed as floats, so no
round-tripping happens at all: the digits in a fixture are the digits Allen
served. Verified by loading both the fixture and the full file and asserting
`np.array_equal(..., equal_nan=True)` over all 69 structures of all 24
experiments.

The row counts above are also recorded as `FIXTURE_COUNTS` in
`fixtures/__init__.py` and checked on every fixture load, the same way the
manifest's counts are checked on every real load: a truncated read hashes
consistently and would otherwise pass every other check.

## What these fixtures are not

They are not a receptor map. 69 structures out of the ~800 an ISH experiment
unionizes, and eight genes out of 42, is a test input and an exploration
aid. Any modelling result has to be computed from the full pinned responses
in `data/expression/`.
