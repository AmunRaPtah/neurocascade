# P21.2: is the multi-drug D2 plasma EC50 table actually obtainable?

Phase 21 rests on one assumption: that a per-drug **plasma concentration at 50%
striatal D2 occupancy** can be sourced for enough antipsychotics to make a drug
registry worth building. Rule 12 says test the load-bearing assumption before
building on it. This document is that test. It contains no ingestion code and
adds no constant to the model; P21.1 and P21.3 are the tasks that spend it.

**Result: six of the eight candidate drugs are reachable in a primary human PET
paper, a seventh is reachable only second-hand, and one is unreachable for a
structural reason. Haloperidol, which the repository already holds, is a ninth.
The threshold of four is met, so Phase 21 proceeds, scoped to the reachable
set.**

## What "reachable" was required to mean

A drug counts as reachable only if a number satisfying all of the following was
**read**, not inferred:

1. a plasma (or serum) **concentration**, in ng/mL or nM, not a dose in mg;
2. fitted as the half-occupancy constant of a one-site model
   `occupancy = Emax x C / (C + EC50)`;
3. in **humans**, measured by PET or SPECT against a D2/D3 tracer;
4. printed in a source that could actually be opened: open-access full text,
   an author manuscript in PMC, a green repository copy, or the free abstract.

A paper that exists but is paywalled with the number absent from its abstract is
**unreachable**, not reachable-with-effort. That distinction is the whole point
of this task.

## The brief's premise was half right, and the half it got wrong matters

The brief said: the pooled per-drug table is in Uchida 2011, PMID 21508857,
J Clin Psychopharmacol 31(3):318-25; it is paywalled at every route; its abstract
prints no ED50 and no concentration unit; so treat the pooled table as
unreachable and go to the primary PET papers per drug.

Both halves of the premise were re-checked here.

**The paywall is confirmed.** `api.unpaywall.org` on 10.1097/JCP.0b013e318218d339
returns `is_oa: false, oa_status: closed` with zero OA locations; Europe PMC
returns `isOpenAccess: N, inPMC: N, inEPMC: N`. The abstract, fetched from
PubMed, prints per-drug **prediction errors** (risperidone n=98, clozapine n=75,
olanzapine n=42, haloperidol n=35, ziprasidone n=31) and no ED50 anywhere. Those
five n values sum to 281, which the abstract also gives as the total, so the
pooled analysis covers exactly five drugs. Quetiapine, amisulpride,
paliperidone and aripiprazole were never in it, and no amount of access to that
paper would have produced them.

**The conclusion drawn from the paywall does not hold.** Rule 11 applies: follow
what the sources actually print. All five of Uchida's ED50 values are recoverable
from open-access secondary sources, in both of the paper's parameterizations:

| Drug | Uchida ED50, unconstrained Emax | Uchida ED50, Emax fixed 100% | Where read |
|---|---|---|---|
| Haloperidol | 0.32 ng/mL | 0.70 ng/mL | Spangemacher 2025, PMC12488982 |
| Olanzapine | 7 ng/mL (a = 90.7%, ED50 7.1) | 10 ng/mL | Spangemacher 2025; Uchida 2014, PMC3960457 |
| Risperidone (active moiety) | 5 ng/mL (a = 88.0%, ED50 4.9) | 8 ng/mL (8.2) | Spangemacher 2025; Uchida 2014; Cuñado 2025, PMC11867924 |
| Clozapine | 105 ng/mL (Emax 60%) | 483 ng/mL | Spangemacher 2025 |
| Ziprasidone | a = 88.2%, ED50 32.9 ng/mL | not seen | Uchida 2014, PMC3960457 |

Uchida 2014 is the same first author restating his own 2011 constants to drive a
CATIE reanalysis; the passage was verified here directly against the PMC HTML
rather than taken from a search summary. Spangemacher 2025 is an independent
group transcribing the same five, and it also records Uchida's own recommendation
to prefer the unconstrained model.

This is worth stating plainly because it changes what "unreachable" means for the
rest of the phase: **the pooled paper is unreachable, its numbers are not.**
What is lost by going through a secondary is not the value but everything around
it: the study list behind each pooled fit, the confidence intervals, the units
convention, and the ability to check whether a fit used data this repository
already holds. Two of those losses are load-bearing, and both are recorded under
"traps" below.

The risk of second-hand transcription is not hypothetical here. Br J Clin
Pharmacol 2022, PMC9197923, Table 2, pairs Uchida's risperidone `Emax = 88%`
with `EC50 = 8.2 ng/mL`. That pair does not exist in the source: 88.0% goes with
4.9, and 8.2 goes with Emax fixed at 100%. A row that mixes the two describes a
hyperbola nobody fitted. Any value taken from a secondary in P21.1 must record
that it is second-hand and must carry its own Emax from the same sentence.

## Per-drug reachability

### Reachable

| Drug | Value read | Basis | Primary source | How it was reached |
|---|---|---|---|---|
| **Risperidone** | ED50 **11.06 ng/mL** (95% CI 9.15-12.96), Emax constrained to 100% | risperidone + 9-OH-risperidone **active moiety**, total plasma | Remington 2006, Am J Psychiatry 163(3):396-401, PMID 16513859 | value is in the free PubMed abstract; full text also read as a green OA repository copy |
| **Olanzapine** | EC50 **11.0 +/- 1.3 ng/mL** | parent, total plasma | Mamo 2008, Neuropsychopharmacology 33(2):298-304, PMID 17443131 | bronze OA; full text read from a repository mirror |
| **Clozapine** | ED50 **950 ng/mL** putamen, 333 ng/mL inferior temporal cortex, Emax "approached 100%" (fitted, not fixed) | parent, total plasma | Gründer 2006, Neuropsychopharmacology 31(5):1027-35, PMID 16237387 | both values printed in the free abstract; body paywalled |
| **Ziprasidone** | ED50 **204 nmol/L (84 ng/mL)**, 95% CI 177-231 nmol/L, R2 0.68, Emax fixed at 100 | parent, total **serum** | Suzuki 2013, Psychopharmacology 228(1):43-51, PMID 23417515 | author manuscript readable in full at PMC3679209 |
| **Aripiprazole** | plasma EC50 **11.1 ng/mL** (95% CI 10.1-12.1); effect-site EC50 8.63 ng/mL with ke0 half-life 0.96 h; Emax fixed at 100 | parent only, total plasma | Kim 2012, J Cereb Blood Flow Metab 32(4):759-68, PMID 22186667 | full text readable at PMC3318151 |
| **Amisulpride** | EC50 **143.6 ng/mL** (putamen) and **59.9 ng/mL** (caudate), with Emax fitted free at 95.7% and 90.1% | parent, total plasma | Vernaleken 2004, Int J Neuropsychopharmacol 7(4):421-30, PMID 15683553 | publisher OA; PDF retrieved through the Silverchair redirect and text-extracted |

Amisulpride is the only drug in the set whose source prints the model equation
in the same table as the constants, `Occupancy = Emax x C / (EC50 + C)` with the
footnote "EC50 is the plasma concentration (ng/ml) predicted to provide 50 %
occupancy of Emax". It is also the only one where a free-fitted Emax comes with
its region, so it is the cleanest row available and a good template for what the
registry should demand of the others.

### Reachable second-hand only

| Drug | Value read | Basis | Attributed primary | Where it was actually read |
|---|---|---|---|---|
| **Paliperidone** | ED50 **6.65 ng/mL** striatum (7.73 ng/mL temporal cortex) | parent paliperidone only, total plasma | Arakawa 2008, Psychopharmacology 197(2):229-35, PMID 18058087 | the sponsor's medical-information page for Invega Sustenna/Trinza, which states the value and cites Arakawa as its reference 6 |

The primary is closed: Springer serves abstract and references only, the free
abstract prints "ED50 values of the striatum and temporal cortex were 2.38 and
2.84 mg/day" in **dose** units and no concentration, and the green-OA record at
the QST repository is metadata with an empty file list, so the OA flag that
OpenAlex and Unpaywall carry for it is false in practice. A manufacturer's
restatement of its own registration study is a weaker provenance class than any
other row here, and it is recorded as such. Corroboration, not confirmation: the
same sponsor's patent US 9,439,906 B2 prints an apparent KD for D2 occupancy of
4.4 to 6.4 ng/mL from two unpublished healthy-volunteer PET studies, so three
human datasets span about 1.5x.

Paliperidone is where the identity trap bites hardest: it *is*
9-hydroxyrisperidone, so almost every other occupancy number for it in the
literature is the risperidone **active moiety** (parent plus metabolite summed).
Arakawa's 6.65 ng/mL is the only pure-paliperidone striatal figure, because his
subjects took paliperidone ER and had no parent risperidone. Pooling it with
Uchida's active-moiety 8.2 would be a category error, not a rounding one.

### Searched and empty

| Drug | Outcome |
|---|---|
| **Quetiapine** | **Unreachable, and the reason is the drug, not the search.** No primary human PET paper prints a plasma EC50, because the fits failed where they were attempted. Vernaleken 2010 (IJNP 13:951-960, PMID 20392299, free full text): "The nonlinear regressions for estimating Michaelis-Menten parameters yielded significant fits for Emax, but did not furnish valid fits for EC50 values in any brain region." Gefvert 2001 (PMID 11313155): "Correlations between receptor occupancy and plasma concentrations of quetiapine could not be calculated." de Greef 2011 excluded quetiapine from its pooled model deliberately. Uchida 2011 never included it. The only readable number, 583 ng/mL, is a review-reported popPK/PD estimate from **SPECT**, attributed by inference rather than by a verified citation, and is not usable as a PET-sourced constant. |

## Why the fit fails for quetiapine, and what that implies for the phase

Quetiapine is the most useful negative result in this task, because it fails for
a reason that is a property of the pharmacology rather than of the literature.
Peak striatal occupancy at clinical doses is roughly 26-60% and falls to
single-digit trough within a dosing interval: Nord 2011 (PMC3198174) measured
trough occupancy of 7 +/- 7% for immediate release and 8 +/- 6% for extended
release at the same 300 mg/day. A one-site model fitted across scans taken at
mixed times post-dose is fitting the scan schedule, not an affinity. Occupancy
also decays more slowly than plasma for essentially every antipsychotic (Kurose
2020, PMID 32726002), which makes a plasma-driven direct-effect model
structurally wrong rather than merely imprecise.

That is the same hysteresis Kim 2012 measured for aripiprazole and resolved with
an effect compartment, which is why that paper reports two EC50 values, 11.1 for
plasma and 8.63 for the effect site. The repository's chain currently has no
effect-site kinetics at all (`docs/validation-ledger.md` lists ke0 as absent, and
the four `HUMAN_KE0_RECORDS` are illustrative). Any drug whose occupancy lags
plasma by hours will inherit that gap as bias, so the P21.6 chain must record per
drug which of the two EC50 conventions its constant came from.

## Traps that must survive into P21.1's registry schema

The brief for P21.1 says the registry's schema is decided by what the sources
actually print. What they print is this:

1. **EC50 and Emax are one measurement, not two.** Every recovered value is
   conditional on whether Emax was fitted or pinned at 100%. Clozapine moves from
   105 to 483 ng/mL on that choice alone, a factor of 4.6. Risperidone moves 4.9
   to 8.2. Haloperidol moves 0.32 to 0.70. A registry that stores an EC50 without
   the Emax it was fitted with stores an ambiguous number. This is the single
   most important schema requirement to come out of this task.
2. **Concentration basis differs by drug and is not optional.** Every reachable
   risperidone value is on the **active moiety** (parent + 9-OH-risperidone);
   no parent-only risperidone EC50 was found anywhere. Every aripiprazole value
   in a primary paper is **parent only**, while Gründer 2008 (via the Hart 2022
   review) reports an active-moiety pair roughly double the parent pair. Feeding
   a summed concentration into a parent EC50, or the reverse, is a silent
   two-fold error, which is exactly what P21.4 exists to prevent.
3. **Region is not "striatum".** Vernaleken 2004 prints amisulpride 143.6 ng/mL
   in putamen against 59.9 in caudate: a **2.4-fold** difference in the same
   nine subjects, with the same tracer, in the same table, and with different
   Emax values too (95.7% vs 90.1%). Gründer 2006 prints clozapine 950
   (putamen), 582 (caudate) and 333 (inferior temporal cortex) ng/mL from one
   dataset. Suzuki 2013 prints 174 (putamen), 205 (caudate) and 190 (ventral
   striatum) nmol/L, and there the caudate is the *higher* one. A single
   "striatal EC50" per drug is a choice from a band up to 2.4x wide, and the
   direction of the caudate-putamen difference is not even consistent across
   drugs. The repository already carries `region=None` on all four points its own
   haloperidol EC50 was fitted on (P19.7); adding drugs whose sources *do* print
   a region while the anchor drug's do not will widen that provenance gap, not
   close it.
4. **Tracer is not interchangeable.** The reachable set spans [11C]raclopride
   (risperidone, olanzapine, aripiprazole, ziprasidone-A, paliperidone),
   [18F]fallypride (clozapine, ziprasidone-B), [18F]desmethoxyfallypride
   (amisulpride) and [123I]IBZM (the quetiapine SPECT value). The fallypride
   family is D2/D3 and cannot separate them, which matters most for amisulpride,
   whose D3 Ki is essentially equal to its D2 Ki: that EC50 is a D2+D3
   composite and must not be labelled pure D2. P19.7 already put the tracer
   spread at a 2 to 2.5x floor larger than every held-out error in the ledger.
5. **Route and regimen differ per drug and are baked into the value.**
   Risperidone's 11.06 is long-acting injectable at steady state; the same
   group's oral value is 6.08, a 1.8-fold difference within one centre using the
   same tracer and assay. Olanzapine's 11.0 pools oral baseline with pamoate
   depot cycles. Aripiprazole's 11.1 is a single oral dose in healthy young men.
   Ziprasidone's 84 is oral steady state with food, and Vernaleken found single
   doses landing outside the steady-state curve's 95% band. Paliperidone's 6.65
   comes from an osmotic-release formulation with a near-flat profile and a
   peak-to-trough ratio around 1.6, which is the regimen least like the ones the
   chain drives. The haloperidol mismatch the repository already carries,
   depot-fitted and orally driven, is the rule here rather than the exception.
6. **The pooled values may not be independent of this repository's own fit.**
   Uchida 2011 pooled 35 haloperidol subjects across published PET studies. Its
   Emax-constrained haloperidol ED50, 0.70 ng/mL, sits 1.2% from the value this
   repository fitted independently in P19.4 (1.8849 nM = 0.708 ng/mL). That
   agreement is striking and it is **not** a validation, because the study list
   behind the pooled fit is in the paywalled paper and cannot be checked for
   overlap with Nyberg 1995 and Regenthal 1997, which are the two studies the
   repository fitted on. Recording it as an independent replication would be
   exactly the error rule 9 is written against. It is recorded in the ledger as
   an unresolved provenance question instead.
7. **Two published numbers were volunteered by web search summarizers and are
   supported by nothing that was read**: a ziprasidone EC50 "in the range of
   5-10 ng/mL" and an olanzapine pair "7 and 10 ng/mL" presented as primary.
   Neither was traced to a source. They are listed here so a later task does not
   rediscover them and mistake them for literature.
8. **One relevant paper is retracted.** Nikisch 2010, J Psychiatr Res
   44:754-759, PMID 20176367, relating quetiapine D2 occupancy to plasma and CSF
   levels, was retracted (notice PMID 22957353) because "PET data of two out of
   these individuals did not show a reliable signal at all". It still returns in
   searches.

## An open-access pooled alternative exists, and it is a different kind of object

de Greef 2011, AAPS J 13(1):107-11, PMC3032087, Table III, is open access and
prints Emax and EC50 for five drugs from one pooled population PK/PD model,
verified here directly from the article HTML:

| Drug | Emax (%) | EC50 (ng/mL) |
|---|---|---|
| Asenapine | 101.8 (4.5%) | 0.528 (0.063) |
| Haloperidol | 92.0 (3.3%) | 0.532 (0.085) |
| Risperidone | 91.2 (3.0%) | 4.43 (0.75) |
| Ziprasidone | 98.0 (10.7%) | 15.4 (5.1) |
| Olanzapine | 87.5 (3.9%) | 5.29 (1.22) |

This is a meta-model fitted to occupancy observations extracted from published
PET studies, not a primary PET measurement, so it stands in the same relation to
the primaries as Uchida 2011 does. It is recorded here because it is the only
multi-drug occupancy table in this whole search that is both open and
first-hand about its own fit, and because its ziprasidone value, 15.4 ng/mL, is
5.5x below Suzuki's primary 84 ng/mL and 2.1x below Uchida's pooled 32.9. Three
pooled or primary analyses of the same drug spanning 5.5x is the honest measure
of how well this quantity is known.

## Where each drug was looked for

Routes used across all eight drugs, with what each contributed:

- **PubMed E-utilities** (`esearch`, `esummary`, `efetch`): the decisive route
  for risperidone (Remington 2006 prints ED50 in the abstract), clozapine
  (Gründer 2006, same) and ziprasidone (Suzuki 2013, same).
- **Europe PMC REST search**, including literal-number queries such as
  `"6.08 ng"` and `"583 ng/mL"` over the full-text index: how the Uchida
  secondary transcriptions were found at all.
- **`pmc.ncbi.nlm.nih.gov/articles/<PMCID>/` HTML**: returned readable full text
  for author manuscripts that Europe PMC's `fullTextXML` endpoint 404s on
  (PMC3679209, PMC3318151, PMC3960457, PMC3032087). Worth remembering: the
  EPMC 404 does not mean the text is unavailable.
- **Unpaywall**: correctly classified Uchida 2011 as closed, found the green
  repository copies of Remington 2006 and Mamo 2008, and identified Vernaleken
  2004 as publisher OA. It is not reliable in the other direction: it reports
  Arakawa 2008 as green OA at the QST repository, and that record turns out to
  carry metadata with an empty file list. An OA flag is a claim to be checked,
  not a result.
- **Europe PMC's `FULL_TEXT:` field query** returned zero hits for both
  amisulpride and paliperidone against terms that are demonstrably in indexed
  articles. Its silence means nothing; unfielded phrase queries worked where the
  fielded ones did not.
- **Publisher sites**: psychiatryonline.org returns HTTP 403; nature.com returns
  an HTML interstitial in place of the PDF for bronze-OA articles;
  tandfonline.com returns 403; academic.oup.com returns a Cloudflare 403 to a
  direct PDF request but serves the file when the redirect to its Silverchair
  watermark host is followed. That redirect is what made the amisulpride row
  possible, and it is worth trying before declaring an OUP article closed.
- **Drugs@FDA**: checked for ziprasidone NDA 20-825, aripiprazole NDA 21-436 and
  paliperidone NDA 21-999. The ziprasidone label prints an in-vitro D2 Ki
  (4.8 nM), not an in-vivo EC50; the original biopharmaceutics reviews are
  scanned and unindexed; the Invega 2006 submission lists a review document in
  its table of contents but every URL variant for it returns 404, and the
  Wayback snapshot of that page was rate-limited rather than read. **No
  FDA-sourced occupancy-concentration relation was found for any drug.** Phase
  21's title calls the phase "FDA-sourced"; on this evidence the EC50 layer is
  not FDA-sourceable and P21.3's PK layer is where the FDA route has to pay off.
  The one loose end worth reopening is that Invega review: if the Wayback copy
  leads to the 2006 Clinical Pharmacology review, paliperidone would move from
  sponsor restatement to auditable public-domain primary.
- **Sponsor and patent documents**, reached only after the peer-reviewed routes
  failed: the Janssen medical-information site and Google Patents both print
  paliperidone numbers that no readable journal article does. They are usable
  evidence of what the sponsor asserts, and nothing stronger.
- **ClinicalTrials.gov**: NCT00934635 has posted results carrying both occupancy
  and plasma outcome measures for paliperidone ER against oral risperidone. That
  is arm-level means rather than a fitted EC50, so it cannot supply a constant,
  but it is an unexplored source of raw pairs if P21.1 ever wants to refit
  rather than transcribe.
- **The local Aqueduct knowledge base**: unavailable throughout, its DuckDB
  warehouse write-locked by the hourly harvest process. It contributed nothing
  to this task and was not retried.

Papers checked and rejected as sources for this quantity, so a later task does
not re-walk them: Nyberg 1999 (PMID 10360125, two fixed doses, no fit), Kapur
1995 (PMID 7543969, dose not concentration), Nyberg 1995 (PMID 7545821, n=5, no
fit), Uchida 2008 (PMID 18642974, means only), Muly 2012 (PMID 22214649,
macaque, disqualified by rule 2), Nordström 1995 (PMID 7573582, explicitly
declines to fit clozapine: "D2 receptor occupancy was low in all patients and was
not described by the curvilinear relationship"), Kessler 2006 (PMID 16738543,
occupancies only), Talvik 2001, Farde 1994, Takano 2006, Attarbaschi 2007
(SPECT, correlation only), Mamo 2004 (PMID 15121646, computes a ziprasidone D2
EC50 but prints only the 5-HT2-to-D2 *ratio* in the abstract; full text closed),
Kapur 1998 (PMID 9659858, the classic olanzapine 10.3 ng/mL; abstract prints
occupancy percentages only and the full text is closed at every route, so the
number is readable only in the Mauri 2014 review, PMC4464358), la Fougère 2005
(PMID 15937316, free full text, amisulpride SPECT n=29, prints only Pearson
correlations of occupancy against dose and against plasma, no fit), Bressan 2003
(psychiatryonline 403, abstract prints no EC50), Meisenzahl 2008 and Sparshatt
2009 (not OA, and the latter is a review), Xiberas 2001 (the amisulpride
temporal-cortex EC50 of 24.1 ng/mL usually attributed to it was in fact
**re-fitted by Vernaleken from Xiberas's data**, with a different tracer and
different subjects, so the striatal-to-limbic ratio that table appears to
support is a cross-study construct and not one measurement).
