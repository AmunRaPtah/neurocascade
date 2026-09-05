# P19.3: Human D2 PET occupancy observations for haloperidol

Status: complete, with the shortfall stated below rather than papered over.

The brief asked for a held-out set of human striatal D2 occupancy measurements
paired with haloperidol dose or plasma concentration, to be used in P19.6 as the
target for the dose-to-occupancy chain built in P19.4 and P19.5. What the search
produced falls short of that in one specific way, and the shortfall shapes every
downstream task, so it is recorded first.

## The headline result of the search

**No per-subject table of oral haloperidol dose against striatal D2 occupancy was
reachable.** The canonical series (Farde 1988, Farde 1992, Nordström 1992,
Nordström 1993, Kapur 1996, Kapur 1997, Kapur 2000) is paywalled at every point
where the per-subject numbers live. Their abstracts print group ranges and, in one
case, a fitted relation. They print no means, no standard deviations and no
per-subject rows. Pirate mirrors were excluded from the search by instruction, so
"paywalled" here means unavailable, with no workaround attempted.

Two per-subject tables were readable in full text. Each is disqualified from the
role the brief wanted, for a different reason:

- **Xiberas 2001** prints four dose groups with plasma levels and a striatal
  measure, using [76Br]FLB 457. The authors call the quantity a *binding index*
  and decline to call it occupancy. The value is registered as
  `quantity="binding-index"` and `compare_pet_occupancy` refuses to difference it
  against a predicted occupancy.
- **Videbæk 2001** prints four subjects with **free** plasma haloperidol and
  [123I]IBZM striatal occupancy, under a 0.7 mg intravenous infusion in healthy
  volunteers. The exposure route and the plasma basis both differ from the oral
  steady-state case the chain is being built for.

Fifteen observations are registered in total. Eight of them carry a central
occupancy value that a prediction can be differenced against, and of those eight,
four are group means from depot studies whose abstracts name no tracer.

## What was registered

`src/neurocascade/human/pet_observations.py`, backed by seven transcription
fixtures under `src/neurocascade/human/fixtures/`. Each fixture is a CSV carrying
one row per printed observation, with a SHA-256 digest recorded in code. The
digest locks our transcription against silent drift. It says nothing about whether
the transcription faithfully reproduces the paper, and
`HumanDatasetManifest.checksum_proves` states that in words so a later reader does
not mistake a green checksum for verified fidelity.

| Dataset | Tracer | n | Points | Quantity | Exposure |
|---|---|---|---|---|---|
| kapur1996 | [11C]raclopride | 7 | 1 | occupancy range 53-74% | 2 mg/day oral |
| kapur1997 | [11C]raclopride | 12 | 1 | occupancy range 53-88% | oral, dose not printed in abstract |
| kapur2000 | [11C]raclopride | 22 | 1 | occupancy range 38-87% | oral, dose not printed in abstract |
| nyberg1995 | not named | 8 | 2 | occupancy 73% and 52% | decanoate depot, weeks 1 and 4 |
| regenthal1997 | not named | 8 | 2 | occupancy 75% and 53% | decanoate depot, weeks 1 and 4 |
| xiberas2001 | [76Br]FLB 457 | per-group | 4 | binding index 0.67-0.94 | 3, 8, 15, 60 mg/day oral |
| videbaek2001 | [123I]IBZM | 4 | 4 | occupancy 16-36% | 0.7 mg intravenous |

The three Kapur entries carry `observed_occupancy=None` with a range instead. A
midpoint was available for the taking and was declined, because no source prints
one. `PETOccupancyObservation.contains()` supports the only test such a point
admits, which is whether a predicted occupancy falls inside the printed interval.

## Two reported relations, kept apart from the observations

Two papers print a fitted dose-to-occupancy relation rather than raw points. These
live in `HALOPERIDOL_D2_REPORTED_RELATIONS`, deliberately separate from the
observation set, because a fitted curve of ours compared against someone else's
fitted curve would be a model-to-model comparison wearing the clothes of a
validation.

- **Kapur 1997**, [11C]raclopride, n=12: EC50 0.51 ng/ml, r² 0.84. The abstract
  also prints 80% occupancy at 2.0 ng/ml. Those two statements agree with each
  other under a rectangular hyperbola: 0.51 ng/ml implies 80% at 2.04 ng/ml. That
  internal consistency is the only check available on a paper whose full text
  could not be reached, and it passes.
- **Fitzgerald 2000**, striatum, n=21: restates a "previously derived" ED50 of
  0.40 ng/ml and reports a prospective mean prediction error of 3.89% (CI
  0.45-7.33), r 0.864.

**These two disagree by 28%.** Fitzgerald cites the Kapur line of work while
printing a different constant. Both are recorded with their printed forms. Neither
is averaged into the other, and nothing in the codebase silently picks one.

## The tracer disagreement that P19.6 will have to report

Videbæk's four subjects each identify their own hyperbola constant, and the paper
prints a Kd column. Reconstructing it from the registered values,
`Kd = C(1-Occ)/Occ`, reproduces the printed 0.16, 0.24, 0.19 and 0.39 nM to two
decimals for all four subjects. A test asserts this, which is a check on the
transcription as well as on the model form.

Those are free-plasma constants. At the free fraction of 9.0% recorded in P19.2, a
mean free Kd of 0.25 nM corresponds to roughly 2.78 nM total, or about 1.04 ng/ml.
Set against Kapur's 0.51 ng/ml and Fitzgerald's 0.40 ng/ml, both total plasma:

| Source | Tracer | EC50, total plasma |
|---|---|---|
| Kapur 1997 | [11C]raclopride | 0.51 ng/ml |
| Fitzgerald 2000 | raclopride lineage | 0.40 ng/ml |
| Videbæk 2001 | [123I]IBZM | ~1.04 ng/ml (converted from free) |

A factor of two to two and a half separates the IBZM-derived constant from the
raclopride-derived ones. This is what the tracer non-interchangeability rule
predicts, and it sets the floor on how well any single-parameter chain can be
expected to match the literature. P19.6 reports it rather than fitting through the
middle of it.

## Studies examined and excluded

Each of these was read far enough to decide, and each is excluded for a stated
reason rather than dropped quietly.

- **Vallabhajosula 1997.** Prints basal-ganglia-to-cerebellum ratios of 1.05 to
  1.61. That quantity moves in the opposite direction to occupancy and lies
  outside [0,1], so it cannot enter `PETOccupancyObservation` at all. Registering
  it as occupancy would have inverted the dose response. The `availability-ratio`
  member of the `quantity` enum exists so that a future attempt to add such a
  value fails at construction.
- **Tuppurainen 2009.** Extrastriatal regions only, n=2 for haloperidol, no plasma
  data. Verified directly rather than taken on report.
- **Kessler 2005**, [18F]fallypride. The only human haloperidol fallypride study
  located. The file retrieved for it is a 564-byte HTML stub rather than a PDF, and
  no other route was found. Unobtainable.
- **Uchida 2011, Lako 2013, PMC11843794.** Meta-analyses or re-digitisations of
  other authors' figures. Lako additionally pools across tracers, which the
  discipline in `pet_validation` forbids.

  The Uchida meant here is **PMID 21508857**, 31(3):318-25, the plasma-levels
  pooled analysis, which is the one that would otherwise have been a candidate
  for haloperidol plasma-occupancy points. Its sibling, PMID 21694629,
  31(4):497-502, is a different paper on clinical effects and is the source of
  the 77-to-78-percent EPS threshold this project does use. Same six authors,
  same journal, same year, confusable titles; P21.2 found the two conflated in
  `occupancy_validation.py` and the disambiguation is now pinned there in
  `UCHIDA_2011_IS_TWO_PAPERS`.
- **"Nordstrom et al. 1993 Biol Psychiatry"**, named as a lead in the phase brief.
  PMID 8097114 is a raclopride study. Citing it for haloperidol would be an error;
  the lead is recorded here so it does not get re-followed.

## Open anomaly

Xiberas reports about 6 ng/ml plasma at 3 mg/day, while Kapur's range for 1 to 5
mg/day oral haloperidol runs 0.5 to 5.8 ng/ml. The Xiberas figure sits at the top
of a range whose dose sits near the bottom. The unit was checked at the source:
`pdftotext` renders the micro sign as `m`, turning µg/l into "mg/l", so page 2 was
rendered at 200 dpi and read as an image to confirm µg/l. The registered nM values
follow from that reading. The discrepancy against Kapur remains unexplained and is
one more reason the Xiberas points stay out of the occupancy set.

## What P19.4 onward inherits

- `occupancy_observations()` returns the eight differenceable points. Four carry
  plasma concentrations on a total basis (the depot means), four on a free basis
  (Videbæk).
- Region is `None` on eleven of fifteen points, because the abstracts that carry
  them name no region.
- Any held-out test at the level the brief imagined, per-subject oral dose to
  striatal occupancy, has no data behind it in this repository. P19.6 tests what
  exists: interval containment against the three Kapur ranges, and a comparison
  against the two reported relations with the tracer disagreement stated.
