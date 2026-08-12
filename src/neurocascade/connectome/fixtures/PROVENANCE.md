# Fixture provenance

Every file in this directory is a **slice of real, published data**, cut
from the checksum-verified cache and written back out in the source file's
own format. None of it is synthetic. Regenerate with:

    python scripts/fetch_connectome.py            # populate data/connectome/
    python scripts/make_connectome_fixture.py     # rewrite this directory

`make_connectome_fixture.py` calls `fetch.verify()` on all five upstream
files before it reads any of them, so a fixture can never be cut from
unverified bytes.

## The region set

Fifteen CCF structures, listed in
`neurocascade/connectome/fixtures/__init__.py` as `FIXTURE_REGION_IDS`:

| id   | acronym | major division | why it is here |
|------|---------|----------------|----------------|
| 985  | MOp     | Isocortex      | cortical partner |
| 972  | PL      | Isocortex      | the `PFC` node of the opioid scenario |
| 56   | ACB     | STR            | the `NAc` node |
| 672  | CP      | STR            | the `Striatum_D2_MSN` node |
| 295  | BLA     | CTXsp          | the `BLA` node of the GABA-A scenario |
| 536  | CEA     | STR            | the `CeA` node of the GABA-A scenario |
| 1022 | GPe     | PAL            | D2 indirect pathway |
| 1031 | GPi     | PAL            | D2 indirect pathway (the saturation lesson) |
| 470  | STN     | HY             | D2 indirect pathway; also the source of the zero-weight connections |
| 362  | MD      | TH             | thalamic partner (`Thalamus` node) |
| 749  | VTA     | MB             | the `VTA_DA` / `VTA_GABA` nodes |
| 381  | SNr     | MB             | basal ganglia output |
| 795  | PAG     | MB             | the `PAG_Interneuron` / `PAG_Output` nodes |
| 872  | DR      | MB             | raphe partner |
| 147  | LC      | P              | the `LC` node; **absent from Oh et al. 2014** |

Eight of the twelve major divisions are represented.

## The five files

### The three Knox et al. 2019 CSVs

Each is 15 sources x 30 targets (15 ipsilateral + 15 contralateral), cut
from the corresponding published file:

| fixture | cut from | sha256 of the published file |
|---|---|---|
| `knox2019_normalized_connection_density_fixture.csv` | `knox2019_normalized_connection_density.csv.gz` | `e202b7262882e91dcb9870a30b372ddecbb3cbcac6b532b60fa1b538ec37aa97` |
| `knox2019_connection_strength_fixture.csv` | `knox2019_connection_strength.csv.gz` | `c2df7d5ac4478bb01c78a06c577c1b58453a0674b4258c070f1e555b9ca5eca8` |
| `knox2019_connection_density_fixture.csv` | `knox2019_connection_density.csv.gz` | `ecd4cc3cfb1ea1a33967de8f1d9183338c512edaaea5fbdb58211a4aa34d9862` |

each of them 291 sources x 577 targets in the published form. Same two-row
header layout as the published files (hemisphere row, then structure-id
row, both with an empty first cell), decompressed because 10 kB of CSV does
not need gzip and a plain-text fixture is reviewable in a diff.

P3.1 needed only the first. P3.2 added the other two because they are what
a network build actually consumes: the raw `connection_strength` is the
only normalization that can be summed across regions, and the ratio
`connection_strength / connection_density` is where the region voxel counts
come from (`connectome/volumes.py`) - a quantity this codebase recovers
rather than writes down. Keeping all three as fixtures means that recovery,
and the check that it lands on whole numbers, runs offline on real data.

**Bit-identical to the published doubles.** Values are written with
`repr(float(...))`, Python's shortest round-tripping decimal form, and were
checked to reload `array_equal` to the corresponding slice of the full file.

### `oh2014_supplementary_table_3_fixture.xlsx`

14 sources x 28 targets (LC has no row or column in Oh et al. 2014), four
sheets in the published order, cut from

    oh2014_supplementary_table_3.xlsx
    sha256 eab971421b9597424d65547e3361e5b9d64c32bc15d21d51b94e5067482c5664
    213 sources x 426 targets

184 of the fixture's 392 p-value cells carry the literal Excel token
`#NUM!`, exactly as the published sheet does, and exactly at the cells whose
weight is zero. That equivalence holds on the full published file too, in
both hemispheres (28415 ipsilateral and 29501 contralateral `#NUM!` cells,
all of them at zero-weight connections and nowhere else), and it is what
the reader relies on to map the token to NaN rather than to zero.

**Round-trips to 16 significant figures, not bit-identically.** openpyxl
serializes floats as `%.16g`, so the largest weight in this fixture differs
from the published double by 4.4e-16 absolute (1 ulp) and the largest
p-value difference is 5.6e-17. That is a property of the xlsx writer, not a
tolerance chosen to make anything pass; the CSV fixture above has no such
gap because CSV lets us write the exact shortest round-tripping form.

### `allen_structure_graph_1_fixture.json`

55 of the 1327 structures: the fifteen fixture regions, all of their
ancestors, all twelve major divisions with their ancestors, and every
structure named by P3.2's sign table (`connectome/neurotransmitter.py`:
RT, ZI, SNc, PALv, PALm, CBX, CBN and the rest) with theirs, in the same
nested `{"msg": [root]}` shape the API returns, cut from

    allen_structure_graph_1.json
    sha256 b5f0d024d1df09ee18aef15093f55240f4a70b3924ee9689add612dadff80386
    1327 structures

Keeping all twelve divisions means `StructureOntology.major_divisions()`
resolves against the fixture, so the division-lookup tests are running the
same code path as production rather than a special case. Keeping every
sign-rule structure does the same for `SignRules`, whose acronym-vs-id
cross-check would otherwise only ever run against the gitignored full
ontology.

## What these fixtures are not

They are not a twelve-or-fifteen-region model of the mouse brain. Slicing
15 regions out of a 291-region whole-brain fit deletes every path that runs
through the other 276, so the sliced matrix is a test input and an
exploration aid, not a scientific object. Any modelling result has to be
computed from the full matrix in `data/connectome/`.
