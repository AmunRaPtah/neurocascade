# P19.2: human haloperidol PK, what was searched and what was found

Searched 2026-09-02. The task asked for a defensible human haloperidol
parameter set, transcribed from a primary source and cited with a resolvable
identifier. The honest answer has two halves.

## The negative half: there is no readable human IV parameter set

Of every human haloperidol study reachable from this host, exactly three
performed a compartmental analysis on **intravenous** data, and none of them
yields a usable micro-parameter set:

| Study | PMID | What it did | Why it does not close the gap |
|---|---|---|---|
| Holley 1983, *Clin Pharmacol Ther* 33(4):477-84 | 6831826 | Fitted "open two- and three-compartment models using nonlinear regression" to IV data in healthy men | Abstract prints only CL 11.8 +/- 2.9 mL/kg/min, Vss 17.8 +/- 6.5 L/kg and the distribution half-lives (0.19 +/- 0.07 h and 2 +/- 1 h). V1, V2 and Q are in the full text, which is behind a Cloudflare-gated Wiley paywall. HTTP 403; a real Chrome session on port 9222 could not clear the Turnstile challenge from this datacenter IP. Not on PMC, not in Europe PMC, not on archive.org. |
| Chang 1992, *Psychopharmacology* 106(4):517-22 | 1579624 | "After intravenous dosing, haloperidol disposition was best characterized by a three compartment model" | Published only CL 21.7 L/h, one distribution-phase volume 1754.3 L, t1/2 and F 0.35 +/- 0.08. The micro-parameters of that three-compartment fit were never printed. Full text closed (Springer). |
| Li 2022, *ICU population PK* | 35335925 | IV data, modern population methods, full text read via PMC8950491 | The authors settled on **one** compartment; sparse sampling could not support a second. CL 51.7 L/h, Vd 1490 L in critically ill patients, a population whose clearance is ~2.4x Chang's. |

Two of the four leads named in the task brief are reviews and cannot supply a
single-study set by construction: Froemming 1989 (PMID 2689040) contributes only
the bioavailability range "60 to 65%", and Kudo & Ishizaki 1999 (PMID 10628896)
contains no PK parameter values at all. It is a metabolism and drug-interaction
reference.

**A propagated-citation warning worth recording.** The widely circulated human
haloperidol values Vd ~18 L/kg and CL ~12 mL/min/kg are commonly attributed to
Forsman & Öhman 1976 (PMID 822989). That record has no MEDLINE abstract and the
paper is not digitised anywhere reachable, so the attribution could not be
checked at source. The numbers are near-identical to Holley 1983's, which is
consistent with the attribution being an error that has been copied forward.
This repository does not use those numbers under either citation.

**Discrepancy that any future transcription inherits.** Holley's Vss (17.8 L/kg)
and Cheng 1987's (7.9 L/kg, PMID 3108922) disagree by a factor of 2.3, and
Chang's F (0.35) disagrees with the Holley/Cheng/Magliozzi cluster (0.65, 0.60,
0.64). Whoever obtains Holley's full text should expect to have to argue for one
of these, not average them.

### Databases and queries

- **PubMed E-utilities**: `esearch`, `esummary` (every PMID above was verified
  against `articleids`, which caught three wrong PMIDs from an earlier pass),
  `efetch`. The decisive query was
  `haloperidol AND ("two compartment"[tiab] OR "two-compartment"[tiab] OR "three compartment"[tiab] OR "three-compartment"[tiab]) AND humans[mh]`,
  which returns **four** records, of which only Holley 1983 and Chang 1992 are
  genuine human haloperidol compartmental analyses.
- **Europe PMC**: `/search?resultType=core` and `/{source}/{id}/fullTextXML`.
  Full text retrieved for PMC9506126, PMC5599451, PMC8950491. Confirmed no full
  text for Holley, Cheng, Froemming, Kudo or Chang.
- **OpenAlex, Crossref, Semantic Scholar, Unpaywall**: no OA location for any
  of those five. Unpaywall and OpenAlex did surface the LMU institutional copy
  of Pilla Reddy 2013, which is how that full text was obtained.
- **Direct publisher fetch**: Wiley 403 plus Turnstile; Springer 303 to
  `idp.springer.com/authorize`.
- **archive.org**: Holley title, `numFound: 0`.
- **CORE was not queried**: `$CORE_API_KEY` is unset on this host. That is a gap
  in the search, not a negative result.

## The positive half: one complete oral apparent set

Pilla Reddy 2013 (PMID 24113674, DOI 10.1097/JCP.0b013e3182a4ee2c) is the only
human haloperidol paper that publishes a complete, internally consistent,
bootstrap-validated two-compartment parameter set. The open-access author copy
at `https://epub.ub.uni-muenchen.de/24066/1/oa_24066.pdf` was downloaded and
Table 2 read directly. Transcribed as printed, original data set column with
%RSE:

| Parameter | Value | %RSE |
|---|---|---|
| CL/F | 88 L/h | 6 |
| Q/F | 233 L/h | 28 |
| Vc/F | 669 L | 29 |
| Vp/F | 2500 L | 39 |
| Ka | 0.236 /h | 18 |

122 subjects (20 healthy volunteers, 102 patients with schizophrenia), 538
plasma concentrations, 1 to 60 mg/day, pooled across seven studies, **oral in
every arm**. The authors: "Haloperidol PK following oral administration was best
described by a 2-compartmental model with first-order absorption."

### Why this is recorded as apparent rather than converted to absolute

No pooled study had an intravenous arm, so F was never estimated and every
volume and clearance is an X/F. This is not a defect for the use P19.5 needs:
F cancels out of an oral plasma-concentration prediction, so the apparent set
predicts the concentration-time course exactly. It does mean V1, V2, Q12 and CL
are individually unidentifiable, and the record says so.

The tempting move is to divide the apparent values by an F from Holley (0.65),
Cheng (0.60) or Magliozzi (0.64) to recover absolute parameters. At F = 0.65
that gives CL ~57 L/h, V1 ~435 L, V2 ~1625 L, Q ~151 L/h, which sits plausibly
between the independent estimates of Franken (29.3 L/h) and Li (51.7 L/h). It is
still a **two-study composite**, it inherits both Pilla Reddy's %RSE of up to 39%
and the F study's SD, and the acceptance criteria forbid assembling one without
labelling it. Since nothing downstream needs it, the record does not assert it.
`HumanPKRecord` enforces this structurally: a record marked `apparent_over_F`
raises if an `F` is set on it.

### Why the values are stored in absolute units, not per-kg

The schema's disposition fields were originally per-kg only. Pilla Reddy's
values cannot honestly go into them. The paper's methods state, verbatim:

> a fixed allometric relationship with individual body weight and CL/F was
> assumed: CL/F = TVCL * [weight / 70]0.75

So clearance scales with an exponent of **0.75**, and the volumes carry no
weight-scaling rule at all. Dividing 88 L/h by 70 kg to get 1.26 L/h/kg and
multiplying it back out for a 90 kg subject applies an exponent of 1 and
over-predicts clearance by about 14%. Individual body weights are not reported
anywhere in the paper: 70 kg is the allometric reference weight, not an observed
mean, and the record labels it as such.

P19.2 therefore added an `absolute` unit family to `HumanPKRecord` alongside the
per-kg one, with `reference_weight_kg` and `cl_allometric_exponent`, and a
`CL_L_per_h_at_weight` method that applies the source's own exponent so nothing
downstream has to guess. A record populates one unit family, never both.

## What is still missing after P19.2

1. No absolute human haloperidol volume or clearance. Obtaining Holley 1983's
   full text through an institutional login is the single highest-value
   unresolved item in this phase.
2. No interindividual variability in the schema. The source reports IIV-CL
   44.5 %CV, IIV-Vc 116 %CV and a proportional residual error of 0.44; the
   record is a typical-subject model with no uncertainty attached, and a 116 %CV
   on central volume is large enough that a single-subject prediction should not
   be read as a population statement.
3. No intramuscular record. The only first-hand IM datum found was Miller 2008's
   median tmax of 37.5 minutes (PMID 18576902), from a non-compartmental study
   in four volunteers, which is not enough for a record.
4. Nothing here has been compared to a measured human observation. That is
   P19.6's job, and it is the only step that can show this set to be wrong.
