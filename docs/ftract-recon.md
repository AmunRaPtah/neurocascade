# P13.1 F-TRACT: reconnaissance already done

Recorded 2026-08-25. This file holds the salvaged output of a P13.1 attempt that
did the acquisition and format work, wrote nothing to the repo, and died on an
OpenRouter daily rate limit at 08:39:35 UTC before it could. Two
earlier attempts that day returned malformed HTTP 200 responses and did nothing
at all, so the backlog recorded three failures and no findings.

Read this before starting P13.1 again. The archive is already downloaded and
checksum-verified. Roughly the first half of the task is done.

## Already on disk, do not re-download

| Path | Size | Digest |
|---|---|---|
| `data/human/ftract/f-tract_v2112.zip` | 915,288,088 bytes | md5 `df03571e8fce5b030482f239dd0d04ab` (matches the digest published on the source record) |
| | | sha256 `9cea8ec2edb4803a7568cf507c1d6f7b208680733beac977f36280a5a6294c46` (computed locally, see licence/digest note below) |
| `data/human/ftract/extracted/ages_15_100/sr_8.40/seg_None_None/pl_200/MNI-HCP-MMP1/` | 16 MB | 1 `.pkl.gz` plus 17 feature CSVs |

`data/` is gitignored in full, so none of this is committable. Only a manifest
and derived fixtures can go in the tree.

The TVB comparison archives were also fetched, to `/tmp/tvb/`:
`connectivity_76.zip` (44,190 bytes), `connectivity_96.zip` (52,641),
`connectivity_192.zip` (61,137). `/tmp` is not durable; re-fetch if gone.

## Provenance

Source used: Zenodo record 7015415, `https://zenodo.org/records/7015415`.
Version 21-12, published 2021-12-09. `f-tract_v2112.zip` is the only file in the
record. The download needed no auth and hit no WAF challenge:

```
curl -sL --retry 3 -o f-tract_v2112.zip \
  "https://zenodo.org/records/7015415/files/f-tract_v2112.zip?download=1"
```

TVB archives came from
`https://raw.githubusercontent.com/the-virtual-brain/tvb-data/master/tvb_data/connectivity/<name>.zip`.

**The EBRAINS DOI named in the task brief (`10.25493/5AM4-J3F`) is not machine
readable.** It 302s to `search.kg.ebrains.eu/instances/ea46bcd6-d19b-479a-8d45-ecc486e3c8b1`,
which is a JavaScript single-page app: an HTTP fetch returns only the page title,
with no file list, no digests, no licence and no version. Do not spend calls on
it.

## Two things that need a decision, not a guess

1. **The licence is contested.** The task brief says CC BY-NC-SA 4.0, citing
   EBRAINS. The Zenodo record page says CC BY 4.0. EBRAINS could not be read to
   settle it. The non-commercial and share-alike clauses differ materially, so
   pick deliberately and record which source the choice came from.
2. **Zenodo publishes md5 only, no sha256.** House manifest style pins sha256.
   The sha256 above is therefore locally computed and verifies nothing about the
   source; the only digest that survives a round trip to the publisher is the
   md5. Record both, and record which one is authoritative.

## Archive layout

The parameter grid has exactly one point per age group. Enumerating every unique
`ages_*/sr_*/seg_*/pl_*` prefix yields only:

```
ages_0_15/sr_8.40/seg_None_None/pl_200/
ages_15_100/sr_8.40/seg_None_None/pl_200/
```

So age group is the only real choice. `sr_8.40`, `seg_None_None` and `pl_200` are
constant across the release, and **what they mean is not documented anywhere in
the archive.** The bundled `scripts/readme.txt` does not define them. The adult
cohort `ages_15_100` was the one extracted.

Under each branch sit 11 parcellations: `MNI-HCP-MMP1`, `AALDilate`, `AICHA`,
`BrodmannDilate`, `Freesurfer`, `Hammers`, and `Lausanne2008-{33,60,125,250,500}`.
The readme also advertises MarsAtlas, which is not present in this release.

Each parcellation directory holds:

```
atlas__<ages>__<sr>__<seg>__<pl>__<parcellation>.pkl.gz
export/<feature>/<feature>.csv        # the full matrix
export/<feature>/<PARCEL>/            # per-parcel subdirectories
```

There are 19 `export/<feature>/` directories and 17 carry a CSV. The two without
one are `synaptic_excitatory_delay` and `synaptic_inhibitory_delay`: **empty
directory entries, no data.** Those two features exist only inside the pickle.
This matters for P13.4, which expects cortical maps of local excitatory and
inhibitory synaptic time constants.

## CSV format and direction semantics

Four comment lines, then the header, then 360 data rows. Every line has exactly
361 comma-separated fields. Missing values are the literal string `nan`.

```
# probability, float64
# MNI-HCP-MMP1 (360 parcels)
# stimulated parcels are listed vertically
# recorded parcels are listed horizontally
stimulated parcels,L_V1,L_V6,L_V2,L_V3,...
L_V1,0.9952095808383233,1.0,0.9846743295019157,...
```

**Rows are the stimulated parcel (source), columns are the recorded parcel
(target).** That is where the directionality lives, and it is stated by the file
itself rather than inferred. Row labels and column labels are the same list in
the same order, which is Glasser's ordering (`L_V1, L_V6, L_V2, L_V3, ...`), not
alphabetical: 180 `L_` parcels then 180 `R_`.

The header is at 0-based line index 4. An earlier parse assumed index 5 and
produced "359 rows, first label L_V6, 179 L_ parcels". That is a parse artifact.
The correct answer is 360 rows, 180 per hemisphere.

## Pickle format

`atlas__*.pkl.gz` is a gzipped pickle of a `numpy.ndarray` with `shape (19,)` and
`dtype object`. Each element is a dict of 360x360 matrices. Keys across the set
are `count`, `mean`, `median`, `median_absolute_deviation`, `std`.

Element index maps to feature in alphabetical order of feature name. This was
established by bit-exact equality against the CSVs (`rtol=0, atol=0`,
`equal_nan=True`), single unambiguous match each:

```
 0 probability                        <- mean, NOT median
 1 peak_latency                       11 dcm_peak_latency
 2 amplitude                          12 axonal_conduction_delay
 3 onset_delay                        13 (no CSV, inferred synaptic_excitatory_delay)
 4 slope                              14 (no CSV, inferred synaptic_inhibitory_delay)
 5 duration                           15 euclidian_distance_peak_velocity
 6 integral                           16 fiber_distance_peak_velocity
 7 onset_latency                      17 euclidian_distance_axonal_velocity
 8 euclidian_distance                 18 fiber_distance_axonal_velocity
 9 fiber_distance
10 goodness_of_fit
```

Every exported CSV is the median except `probability`, which is the mean.

**Indices 13 and 14 are inferred from alphabetical position and are not
verified.** They have no CSV to check against. Their keys are `count`, `median`,
`median_absolute_deviation`. If P13.4 depends on which is excitatory and which is
inhibitory, verify it against the paper before relying on it.

Units are not declared for any feature. The CSVs say only `float64`. Do not
assume milliseconds without a source.

## Measured directionality and coverage

Computed from the on-disk CSVs, NaN-aware, adult cohort, MNI-HCP-MMP1. The
identity `2 x reciprocal + one-way + diagonal = finite` holds for each, which is
the arithmetic check that these are counted consistently:

| | probability | amplitude | axonal_conduction_delay |
|---|---|---|---|
| finite cells of 129,600 | 27,066 (20.9%) | 16,050 (12.4%) | 16,050 (12.4%) |
| on the diagonal | 295 | 278 | 278 |
| reciprocal pairs (measured both ways) | 10,789 | 6,176 | 6,176 |
| of those, asymmetric | 8,439 (78.2%) | 6,176 (100%) | 6,175 (100%) |
| measured in one direction only | 5,193 | 3,420 | 3,420 |

`np.array_equal(M, M.T)` is `False` for all three, and 78.2% of reciprocal
probability pairs differ between directions. That asymmetry is the point of the
dataset and the substance of acceptance criterion 2.

NaN means the ordered pair was never probed; a value means it was measured. For
`probability`, the count of finite cells equals exactly the count of cells with
`count > 0`. 28 measured cells are exactly 0.0, so zero and missing are
different states and must not be conflated. `count` reaches 1951 for probability
and 1191 for the two synaptic elements. The diagonal is populated, so
within-parcel stimulation and recording pairs exist and need an explicit
decision rather than silent inclusion.

**Any coverage or asymmetry test must mask with `np.isfinite`.** `NaN != 0`
evaluates true in numpy, so `np.sum(M != 0)` reports 129,600 "nonzero" cells for
amplitude when 113,550 of them are NaN. The figures above replace an earlier set
computed that way.

## TVB negative control: the task brief is wrong

Acceptance criterion 4 wants a test that rejects the TVB ordinal matrices. The
brief states that `connectivity_76`, `_96` and `_192` all carry ordinal
`{0,1,2,3}` weights and `RM-` prefixes. Measured from the real published files:

| | shape | unique weights | label style | symmetric |
|---|---|---|---|---|
| `connectivity_76` | (76, 76) | 34 continuous values | `rA1`, `rA2`, `rAMYG` | no |
| `connectivity_96` | (96, 96) | `{0, 1, 2, 3}` | `RM-TCpol_R`, prefixes `BG`/`MM82a`/`RM`/`TM` | no |
| `connectivity_192` | (192, 192) | 34 continuous values, identical set to `_76` | `lAD`, `lAM`, `lCAUD` | no |

**Only `connectivity_96` matches the brief's description.** A rejection test
written against the brief as stated would fail to reject two of the three files
it is meant to catch. Test on several signatures instead: shape is not
(360, 360); labels are not HCP-MMP1 Glasser; the value set is quantized or
ordinal; labels carry CoCoMac prefixes.

`info.txt` is identical in all three and reads `weights_unit = "au"`, so TVB's
own metadata concedes the weights are arbitrary units rather than a physical
measurement. That is the cleanest single argument for the rejection.

`tvb-data` is GPL v3 (confirmed via the GitHub licence API), so the zips cannot
be vendored as fixtures. Pin the raw URLs plus digests in the manifest as
verification-only entries, following the outer-bundle and inner-member pattern
already used for the CIT168 bundle in `src/neurocascade/human/midbrain.py`, and
commit only small derived signature fixtures (shape, sorted unique values,
verbatim label list).

## Traps

1. `python` is not on PATH. Use `/root/projects/pardalos/.venv/bin/python`.
2. Shell cwd resets between some tool calls. Use absolute paths.
3. `__MACOSX/._*` shadow entries and `.DS_Store` files litter every directory of
   the zip. Filter them from every listing and extraction.
4. `connectivity_192.zip` nests members under `connectivity_192/` while `_76` and
   `_96` put them at the zip root. Match member paths by suffix, not by literal.
5. Labels contain hyphens and digits (`L_OP2-3`, `L_p9-46v`, `R_a9-46v`) but no
   commas. Splitting on commas is safe; splitting on hyphens is not.
6. Unpickling emits a numpy `DeprecationWarning` about `numpy.core.numeric`. It
   is harmless.
7. Never checksum a file mid-download. The transcript contains an md5 computed at
   551,419,904 bytes which is meaningless.

## Suggested restart

Acquisition and format decoding are done. What remains:

1. Pin the citation facts from Lemarechal et al., Brain 2022;145(5):1653-1667,
   PMID 35416942, DOI `10.1093/brain/awab362`: 780 patients, 34,354 stimulations,
   774,445 CCEPs, and the median axonal conduction delay and velocity that P13.4
   quotes.
2. One check of `f-tract.eu/atlas` for a parcellation-specific download, so the
   manifest need not pin a 915 MB archive for a 16 MB subset. If there is none,
   pin the Zenodo zip as the outer bundle with inner-member digests.
3. Write `src/neurocascade/human/ftract.py` and its manifest, imitating
   `midbrain.py`, plus `scripts/fetch_human_ftract.py` and a fixture builder.
4. Feature shortlist for ingest: `probability` (mean), `amplitude` (median),
   `axonal_conduction_delay` (median). The last is the one that feeds
   Wilson-Cowan propagation delays.
5. Fixtures must be verbatim slices. 17 CSVs at roughly 700 KB each is too large
   to commit.
6. `tests/test_human_ftract.py`: asymmetry assertion, the four sampling
   limitations from the brief, and the multi-signature TVB rejection above.
7. Phase 13 section in `docs/architecture.md`, full suite, commit.
