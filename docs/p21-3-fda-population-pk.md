# P21.3: human population PK per antipsychotic, from FDA review documents

Phase 21 wants a per-drug dose→plasma layer to sit underneath the per-drug
occupancy layer P21.2 scoped. This task sources that layer from the US FDA
rather than from the literature, on the brief's premise that the Clinical
Pharmacology review PDFs carry exactly the population-PK analyses the paywalled
journals withhold.

**Result: 17 fitted parameter sets transcribed across 7 drugs and 16
applications, 6 applications recorded as searched-and-empty under a four-way
taxonomy of *why*, and 13 of the transcribed sets compared against a printed
label half-life: 5 within twofold, 8 outside. The failures are the result.
They are concentrated, they have a single shared cause, and one depot that
passes is what proves the cause.**

The comparison is in `half_life_comparisons()`; the records are in
`src/neurocascade/human/fda_popk.py`; the ledger row is V5.

## Two claims already in this repository were wrong, and this task disproves both

Rule 11 says the brief may be wrong about structure. Here it was not the brief
that was wrong. It was us.

**Claim 1, `docs/validation-ledger.md` line 76 (P21.2):** "the Invega clinical
pharmacology review 404s at every URL variant."

It does not 404. `NDA 021999` fetches fine once the TOC extension is right (see
below). What it does is worse and more interesting: the population-PK section is
**withheld**, thirteen consecutive pages replaced by a full-page "Trade Secret /
Confidential" pull-sheet. A 404 is a routing failure you can retry around. A
redaction is a permanent, deliberate, sponsor-requested absence. The two look
identical from a script and are opposite facts. `_PALIPERIDONE_INVEGA_ORAL` is
registered as `searched-and-empty` with `extraction_barrier="redacted"` for
exactly this reason, and its note carries the string "does not 404" so the old
claim cannot quietly survive.

**Claim 2, `src/neurocascade/human/fda_sources.py`:** review documents are "not
machine-fetchable: they are PDFs behind a per-application browse UI."

26 of them were fetched, non-interactively, in a loop. The corrected comment
records the method.

## How Drugs@FDA is actually reached

Four things had to be right at once, and each of them fails silently:

1. **The table-of-contents page is `.html`, not `.cfm`.** The browse UI links to
   `index.cfm?event=overview.process&ApplNo=NNNNNN`, which is a CGI entry point,
   but the per-year document index it eventually reaches is a static file at
   `drugsatfda_docs/nda/<year>/<...>toc.html`. Requesting `.cfm` there returns a
   page rather than an error, which is why this looked like a browse-only UI.
2. **Do not send a browser User-Agent.** A `Mozilla/5.0` string trips the FDA's
   WAF, which answers 302 to `abuse-detection-apology.html`. curl's *default*
   agent is served normally. This is the reverse of the usual scraping folklore
   and cost the most time.
3. **Match the TOC's PDF links case-insensitively.** `ClinPharmR`, `BioPharmR`,
   `MultidisciplineR` and `IntegratedR` appear in several casings across years.
4. **Newer TOCs have no literal links at all.** They synthesise filenames in
   JavaScript from a `var pdfBaseName`, so a link-extracting fetch of a 2022
   application returns nothing while the PDFs sit at predictable URLs.

## The barrier that actually blocks extraction is not redaction

The working assumption going in, inherited from P21.2's "scanned and unindexed",
was that the missing numbers are redacted. Across 26 reviews that turned out
to be true exactly once (NDA 021999). The dominant barrier is that **sponsor
final-parameter tables are raster images pasted into an otherwise text-native
PDF.** `pdftotext` returns the caption and the surrounding prose and silently
skips the table. Nothing signals that anything was skipped.

Two recovery routes both work: `pdftoppm -r 300 -gray -png` into
`tesseract --psm 6`, or rendering the page to PNG and reading it directly.

Two further traps were found by doing this:

- **The caption and the table can be on different pages.** For LYBALVI
  (NDA 213378) the caption is at the bottom of PDF page 284 and the table is at
  the top of 285. A caption-anchored extractor returns an empty table and the
  result is indistinguishable from a redaction.
- **A text layer can be present and wrong.** For ARISTADA INITIO (NDA 209830)
  `pdftotext` returns a plausible table with digits missing from individual
  numbers. Every value in `_ARIPIPRAZOLE_ARISTADA_INITIO` was read visually from
  a render instead, and the record says so.

So the registry does not have a single `unreachable` flag. `extraction_barrier`
takes one of five values, because the five have different futures:

| barrier | meaning | recoverable? |
|---|---|---|
| `redacted` | full-page Trade Secret pull-sheet | no, permanently |
| `image-table` | raster table, part recovered, part not | yes, by OCR or rendering |
| `not-fitted` | no population-PK analysis was performed | no, nothing exists to find |
| `estimates-not-printed` | analysis performed and validated, no table printed anywhere in the review | no, from this source |
| `no-review-document` | application has no review PDF posted | maybe, via FOIA |

Four of the five are exercised by a real row. `not-fitted` accounts for four of
the six empty applications, including both of the ones the brief most expected
to succeed: **Abilify oral (NDA 021436) and Seroquel XR (NDA 022047) never had a
population-PK analysis done at all.** The oral originals are the old
applications; population PK entered this therapeutic area with the depots.

## Apparent versus absolute is decided by a sentence, not by a symbol

P19.2's rule was carried over: if bioavailability was never identified, the
values are apparent and passing a separate `F` is forbidden. Applying it here
exposed that **the symbol printed in the table does not decide the question.**

- **NDA 207946 (INVEGA TRINZA)** prints bare `CL` and `V`. The table legend
  nonetheless states the estimates are conditioned on a fixed relative
  bioavailability, so they are apparent. Read the symbol, and this row is
  wrongly absolute.
- **NDA 022264 (INVEGA SUSTENNA)** prints bare `CL` and `V` and *is* absolute,
  because the analysis was anchored to an intravenous reference; the review
  prints the absolute `F` of 27.7% (90% CI 23.4–32.8, n=20, study P01-0007)
  separately.
- **NDA 209510 (BARHEMSYS)** is intravenous, so absolute by route.

Every record therefore carries `apparent_parameters` **and**
`apparent_evidence`, a quoted phrase from the document that decided it. A test
asserts that an explicit `/F` symbol can never sit in a record flagged absolute,
while a bare symbol is permitted in an apparent one. The asymmetry is the point.

Paliperidone appears in this registry both ways, from two applications of the
same molecule. That is not an inconsistency to be resolved; it is what the two
reviews each did.

## Why the closing result is a half-life comparison

Rule 9 requires a comparison against a number nobody here chose. The
parameters themselves cannot supply one, because they are transcriptions and comparing
a transcription to its source tests nothing but the typing.

The available external number is the **terminal half-life printed in the
approved label**, which is a different surface (`api.fda.gov/drug/label.json`,
DailyMed) from the review PDFs and is generally derived by non-compartmental
analysis of the pivotal studies rather than from the population model.

The comparison is legitimate specifically because **half-life is a ratio.** For
an apparent parameter set, `V/F` over `CL/F` cancels the unidentified `F`
exactly. A half-life computed from apparent parameters is not itself apparent.
That is the one quantity that can cross the apparent/absolute line without an
assumption, which is why it and not `CL` is what gets compared.

`terminal_half_life_h()` computes it properly for the multi-compartment rows: it
builds the rate-constant matrix, solves for the hybrid eigenvalues (the cubic in
closed form), and takes `ln2` over the smallest-magnitude root. It does not use
the `0.693·V/CL` shortcut, which is only correct for one compartment.

### The result

| application | analyte | formulation | predicted t½ (h) | label t½ (h) | fold | terminal AUC frac | cmt | |
|---|---|---|---|---|---|---|---|---|
| NDA 213586 | parent | UZEDY | 3.21 | 3 | 1.07 | 1.000 | 1 | PASS |
| NDA 213586 | metabolite | UZEDY | 11.48 | 21 | 1.83 | 1.000 | 1 | PASS |
| NDA 020592 | parent | ZYPREXA oral | 37.38 | 21–54 | 1.00 | 1.000 | 1 | PASS |
| NDA 213378 | parent | LYBALVI | 49.83 | 21–54 | 1.00 | 0.701 | 2 | PASS |
| NDA 022173 | parent | ZYPREXA RELPREVV | 34.33 | 21–54 | 1.00 | 1.000 | 1 | PASS |
| NDA 022264 | parent | INVEGA SUSTENNA | 48.84 | 23 | 2.12 | 1.000 | 1 | FAIL |
| NDA 202971 | parent | ABILIFY MAINTENA | 168.02 | 75 | 2.24 | 0.223 | 3 | FAIL |
| NDA 217006 | parent | ABILIFY ASIMTUFII | 168.02 | 75 | 2.24 | 0.223 | 3 | FAIL |
| NDA 209510 | parent | BARHEMSYS | 13.58 | 4–5 | 2.72 | 0.268 | 3 | FAIL |
| NDA 216352 | parent | ERZOFRI | 293.81 | 23 | 12.77 | 1.000 | 1 | FAIL |
| NDA 207946 | parent | INVEGA TRINZA | 353.79 | 23 | 15.38 | 1.000 | 1 | FAIL |
| NDA 207533 | parent-and-metabolite | ARISTADA | 4221.57 | 75 | 56.29 | 0.180 | 2 | FAIL |
| NDA 209830 | parent | ARISTADA INITIO | 12296.15 | 75 | 163.95 | 0.050 | 2 | FAIL |

`fold` is 1.00 by definition when the prediction falls inside a label interval.

Per rule 5, no tolerance was widened. Twofold is the threshold that was set
before the numbers were computed and it is the threshold reported against.

### What the failures mean

They are not scattered. Eight of thirteen fail, and every failure is a **depot
or extended-release formulation except one**, while every immediate-release row
passes. The mechanism is flip-flop kinetics: for a depot the terminal slope of
the plasma curve is the *release* rate from the injection site, not the
elimination rate. NDA 022264's review says this in as many words. The label
half-life for these products is therefore describing a different physical
process from the one `ln2/λ_z` of the disposition model describes, and the two
were never the same number.

`terminal_phase_auc_fraction()` is the diagnostic that separates a real terminal
phase from an arithmetic artefact. It reconstructs the exponential coefficients
`A_i` from the eigenvalues and returns the share of total AUC carried by the
slowest phase. The two worst failures are exactly the two rows where that share
collapses: **ARISTADA INITIO at 0.050 and ARISTADA at 0.180.** Those fits parked
the prodrug release rate in a *peripheral compartment*, so the model's slowest
eigenvalue is a release constant wearing a distribution compartment's clothes,
contributing 5% of the exposure and 164-fold of the error. A half-life that
carries 5% of the AUC is not a half-life anyone would measure.

**ZYPREXA RELPREVV is the row that keeps this honest.** It is a depot, a
pamoate intramuscular suspension with a three-week release, and it *passes*, at
fold 1.00. It passes because its review modelled release explicitly, as a mixed
zero-order/first-order input with its own `D3` and `F2`, instead of letting a
peripheral compartment absorb it. The disposition parameters left over are then
real disposition parameters and their ratio is a real half-life. So the failure
is not "depots cannot be checked this way"; it is "a fit that hides release
inside disposition cannot be checked this way," and the AUC fraction tells you
which kind you are holding before you look at the answer.

**BARHEMSYS is the one failure attributable to neither cause.** It is
intravenous, absolute, three-compartment, and still misses 2.72-fold high
(13.58 h against a 4–5 h label). The likely explanation is the sampling window:
a 24 h protocol resolves the dominant phase (2.6 h here) and leaves the true
terminal phase (27% of AUC) poorly identified, so the label's non-compartmental
number and the model's slowest eigenvalue are describing different phases of the
same curve. This is reported, not repaired.

### How independent is this comparison, really

Less than fully, and the caveats belong on the row rather than in a footnote:

- **Four of the five label sources are ANDA generics**, whose labels are copies
  of the innovator's. They are still externally printed numbers that this
  repository did not choose, but they are not independent measurements. Only
  amisulpride's label is the innovator's, and `LabelHalfLife.innovator` records
  which is which; a test asserts exactly one is flagged.
- **Amisulpride is partially circular.** The Barhemsys label's clinical
  pharmacology section quotes clearances from the very population model being
  tested. The half-life itself is non-compartmental, but the same review is
  behind both sides of that one row.
- Label half-lives are frequently rounded to one significant figure ("75
  hours"), so a fold error near 1 is not evidence of precision.

## What is still missing

- **Ziprasidone and clozapine have no transcribed row.** Clozapine's VERSACLOZ
  (NDA 203479) is `not-fitted`; the older clozapine and ziprasidone applications
  predate routine electronic posting.
- **Quetiapine remains empty from both directions.** P21.2 found no plasma EC50
  for it because of its occupancy time course; P21.3 finds no population PK for
  it because NDA 022047 never fitted one. The two gaps are unrelated and the
  drug is out of reach on both layers.
- **None of this is wired into the dose→occupancy chain.** `_require_pk_record`
  in `dose_to_occupancy.py` is haloperidol-only by construction and refuses
  anything that is not two-compartment, absolute, and single-analyte. Most of
  this registry is none of those things. Connecting them is a later task and
  will require deciding what the chain does with an apparent parameter set,
  which is a modelling decision, not a transcription.
- **Inter-individual variability is transcribed but unused.** Every row that
  printed IIV carries it, with scale (`CV%`, `variance`, `SD`) and shrinkage
  where given, because the scale convention differs between reviews and guessing
  it wrong is a factor-of-anything error. Nothing consumes it yet.
