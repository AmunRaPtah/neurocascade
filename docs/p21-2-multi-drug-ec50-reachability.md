# P21.2: can the multi-drug plasma EC50 table actually be obtained

Phase 21 rests on one assumption: that a per-drug plasma EC50 for striatal D2
occupancy can be sourced for enough antipsychotics to make a multi-drug chain
worth building. This document tests that assumption before any ingestion code
exists, and records what is reachable per drug, what is not, and where each
answer was looked for.

The threshold was set in advance. Fewer than four drugs reachable means
`BLOCKED_ON_DATA` and no registry. **Eight of nine are reachable.** The
assumption holds and Phase 21 may proceed, but not in the shape the phase brief
assumed. See "The pooled table does not exist in the form the brief assumed"
below.

No ingestion code was written in this task. Nothing here is loaded by the
model. The next task that wants these numbers must transcribe them from this
document with the caveats attached, or refit them from the per-subject tables
noted below.

## The rubric, fixed before searching

Adopted before any search ran, so that "reachable" could not be redefined by
whatever happened to turn up.

- **REACHABLE_PRINTED**: a primary human PET or SPECT paper prints an
  EC50/ED50/IC50/Ki,app in a concentration unit, and that sentence was read
  first-hand in text retrieved over a free route.
- **REACHABLE_FITTABLE**: free full text contains a table of per-subject or
  per-scan plasma-concentration and occupancy pairs, so the constant can be
  fitted in this repository the way P19.4 fitted haloperidol's.
- **SECONDARY_ONLY**: the value is readable only in a review, pooled analysis,
  patent, or manufacturer page that quotes a primary paper whose own text could
  not be retrieved.
- **UNREACHABLE**: paywalled at every free route tried, or nonexistent.

Figures are not fittable. A scatter plot is not a table. Pirate mirrors were
out of scope and none was queried. Every value below was re-verified by this
session against the retrieved text rather than accepted from a search summary.
Two search summaries volunteered numbers that did not survive that check, and
both are recorded under "What was rejected, and why" rather than deleted.

## Per-drug reachability

| Drug | Verdict | Value as printed | Basis | Primary source |
|---|---|---|---|---|
| Haloperidol | already in repo | 1.8849 nM (0.708 ng/mL) | total plasma, fitted in-repo | P19.4, not re-sourced here |
| Risperidone | REACHABLE_PRINTED and REACHABLE_FITTABLE | ED50 11.06 ng/mL (95% CI 9.15–12.96) | total plasma, **active moiety** (risperidone + 9-OH) | Remington 2006, PMID 16513859 |
| Olanzapine | REACHABLE_PRINTED | EC50 11.0 ± 1.3 ng/mL | total plasma | Mamo 2008, PMID 17443131 |
| Aripiprazole | REACHABLE_PRINTED | EC50 11.1 ng/mL (10.1–12.1) PD-only; 8.63 ng/mL (7.75–9.51) PK-PD | plasma, parent only | Kim 2012, PMID 22186667 |
| Ziprasidone | REACHABLE_PRINTED | ED50 204 nmol/L (84 ng/mL), CI 177–231 | total **serum** | Suzuki 2013, PMID 23417515 |
| Quetiapine | REACHABLE_PRINTED | Ki,app 1369 nmol/L parent; 2004 nmol/L active moiety | total plasma | Nord 2011, PMID 21477416 |
| Clozapine | REACHABLE_PRINTED | ED50 950 ng/mL putamen; 333 ng/mL inf. temporal cortex | plasma | Gründer 2006, PMID 16237387 |
| Amisulpride | REACHABLE_PRINTED (elderly AD) and REACHABLE_FITTABLE (schizophrenia) | IC50 19.1 ng/mL caudate, 61.3 ng/mL putamen | total whole blood | Reeves 2017, PMID 28858444; la Fougère 2005, PMID 15937316 |
| Paliperidone | **SECONDARY_ONLY** | ED50 6.65 ng/mL striatum, not read from primary text | total plasma, parent only | Arakawa 2008, PMID 18058087, paywalled |

Eight drugs are reachable at primary strength and one is not. The count clears
the four-drug threshold twice over, so this task closes as a positive result,
with paliperidone recorded as searched-and-empty rather than defaulted.

### Risperidone, the strongest source in the set

Remington G, Mamo D, Labelle A, Reiss J, Shammi C, Mannaert E, Mann S, Kapur S.
*Am J Psychiatry* 2006;163(3):396–401. PMID 16513859, DOI
10.1176/appi.ajp.163.3.396. [11C]raclopride, striatum, n=9 patients with
schizophrenia or schizoaffective disorder, long-acting injectable risperidone
25/50/75 mg every two weeks at steady state, two scans per patient.

The paper prints the model this repository already uses,
`occupancy = alpha * C / (C + ED50)` with Bmax constrained to 100%, and the
fitted ED50 with an interval. It is also the only source in the set that
carries a per-scan table: Table 1 gives 9 patients by 2 time points, so
**18 (concentration, occupancy) pairs**. This constant can therefore be
refitted in-repo rather than transcribed, exactly as P19.4 did for haloperidol.
That matters for the reason it mattered there. A transcribed constant cannot be
checked against its own data; a refitted one can.

Route: `https://www.um.edu.mt/library/oar/bitstream/123456789/23581/1/A_PET_Study_Evaluating_Dopamine_D_2_Receptor_Occup.pdf`,
HTTP 200, 135,997 bytes, published version, University of Malta institutional
repository. The value also appears in the free PubMed abstract.

**The basis is the active moiety, not parent risperidone.** Mixing this value
with a parent-only concentration is a silent order-of-magnitude error, and it
is the same active-moiety question P21.4 exists to handle.

Two other primary human values exist and disagree. The same group's earlier
oral-risperidone letter (PMID 9472848) gave 6.08 ng/mL on n=9 points, and
Kodaka 2011 (PMID 20860881) gave 9.15 ng/mL with [11C]MNPA and 8.01 ng/mL with
[11C]raclopride in 11 healthy men. Neither full text is free, so neither was
read at primary strength, and the Kodaka concentration basis could not be
confirmed as parent-only. The honest spread across primary human sources is
roughly 6 to 11 ng/mL.

### Olanzapine

Mamo D, Kapur S, Keshavan M, Laruelle M, Taylor CC, Kothare PA, Barsoum P,
McDonnell D. *Neuropsychopharmacology* 2008;33(2):298–304. PMID 17443131.
[11C]raclopride, caudate and putamen, n=14 patients, oral olanzapine
15.2 ± 4.8 mg/day at baseline and then olanzapine pamoate depot 300 mg IM every
four weeks for six months, with trough-proximal scans.

Printed: "an EC50 of 11.0±1.3 ng/ml (mean±SE)". The paper labels the same
dispersion "±SD" in its Figure 1 legend, so the interval type is internally
inconsistent in the source and has to be recorded as ambiguous rather than
resolved by picking one. **Emax is defined in the equation but no numeric
estimate is printed**, so the hyperbola's asymptote is missing from the record.

Route: `https://www.um.edu.mt/library/oar/bitstream/123456789/23614/1/D2_Receptor_Occupancy_of_Olanzapine_Pamoate_Depot_.pdf`,
HTTP 200, 110,926 bytes. Not fittable: Table 1 gives occupancy and
concentration as separate group means with **different n per row** (week 6 has
occupancy n=0 against plasma n=1), so the rows are unpaired.

The fit pools an oral baseline scan with depot scans, which repeats the
oral-versus-depot regimen mismatch already recorded as the largest unrepaired
mismatch in the haloperidol chain.

### Aripiprazole

Kim E, Howes OD, Kim BH, Jeong JM, Lee JS, Jang IJ, Shin SG, Turkheimer FE,
Kapur S, Kwon JS. *J Cereb Blood Flow Metab* 2012;32(4):759–68. PMID 22186667,
PMCID PMC3318151, free full text. [11C]raclopride, ROI "the striatum (putamen
and caudate nucleus)" with cerebellum reference and a simplified reference
tissue model, n=18 volunteers, serial plasma sampling.

This paper is unusually useful here because it prints two EC50 values for the
same data and says why they differ: 11.1 ng/mL from conventional PD modelling
alone, 8.63 ng/mL from combined PK-PD modelling that accounts for hysteresis
between plasma concentration and occupancy. The 1.29-fold gap between them
measures the error introduced by the instantaneous plasma-to-occupancy mapping
this repository's chain currently makes. Whichever value P21.3 takes, the other
one is the size of that modelling choice.

Corroboration, all secondary: Gründer 2008 reports 10 ± 4 ng/mL in putamen, a
figure read only inside Kim's discussion and not from Gründer's own text. Hart
2022 (PMC9020768) tabulates three primary aripiprazole studies at 9 to 12 ng/mL
in striatal regions, consistent with Kim. Horowitz and Taylor 2023
(PMC10501077) refit [18F]fallypride data themselves and obtained 17.24 ng/mL
with Emax 100.2%, which sits well outside that cluster and shows how much the
result depends on tracer and on who does the fitting.

Aripiprazole is a D2 partial agonist. A sourced EC50 makes its **occupancy**
predictable and says nothing about the effect transducer, which is what P21.5
exists to address.

### Ziprasidone

Suzuki T, Graff-Guerrero A, Uchida H, Remington G, Caravaggio F, Borlido C,
Pollock B, Mulsant B, Deluca V, Ismail Z, Mamo D. *Psychopharmacology (Berl)*
2013;228(1):43–51. PMID 23417515, PMCID PMC3679209. [11C]raclopride, putamen,
caudate and ventral striatum, n=11 scanned patients, oral 60 mg twice daily at
steady state, scans at roughly 5, 13 and 23 h post-dose.

Printed in the free abstract and in the full text: 204 nmol/L (84 ng/mL), CI
177–231, R² 0.68. By region, putamen 174, caudate 205, ventral striatum
190 nmol/L.

Four caveats belong with the number. The basis is **serum**, though the paper
writes "plasma level" inside its own equation. Emax was **fixed at 100%**
rather than fitted. Negative occupancy observations were **excluded from the
fit**, a censoring that biases the estimate upward. And occupancy was
referenced to a historical control group of 44 subjects rather than to each
patient's own drug-free baseline.

The single fixed dose means the fit is driven by a narrow concentration range.
Two other primary studies at wider dose ranges reported substantially lower
values: Mamo 2004 at 33 ng/mL and Vernaleken 2008 at 39 to 64 ng/mL. Both are
paywalled and readable only through Suzuki's own discussion. The primary spread
for ziprasidone is therefore about 2.5-fold, the same order as the tracer floor
already recorded in the ledger.

### Quetiapine

Nord M, Nyberg S, Brogren J, Jucaite A, Halldin C, Farde L. *Int J
Neuropsychopharmacol* 2011;14(10):1357–66. PMID 21477416, PMCID PMC3198174,
open access. [11C]raclopride on HRRT, putamen, n=11 healthy males aged 21 to
29, oral once-daily 300 mg at steady state, scans at Tmax and at trough.

Printed: Ki,app 1369 nmol/L for parent quetiapine pooled across immediate- and
extended-release, 1373 for IR alone, 1365 for XR alone, and 2004 nmol/L for the
active moiety when quetiapine and norquetiapine are summed on the assumption of
equal D2 affinity. The paper prints its own conversion anchor, "<0.70 ng/ml
(corresponding to <1.8 nmol/l quetiapine)", so a ng/mL figure is derivable from
source-printed values rather than invented.

Three flags travel with this one. **Occ_max was fixed to 100%** rather than
fitted. The subjects are **healthy volunteers**, not patients. And the authors
state plainly that once-daily IR dosing "can therefore not be directly
translated to the more commonly used twice-daily dosing".

The fourth flag is the most important, because it contradicts the premise that
a quetiapine EC50 is a well-defined quantity at all. Vernaleken 2010 fitted the
same model to 16 patients with [18F]fallypride and reported that it failed:
"The nonlinear regressions for estimating Michaelis-Menten parameters yielded
significant fits for Emax, but did not furnish valid fits for EC50 values in
any brain region." Nord obtained a stable constant only because Emax was fixed
by fiat. Any use of 1369 nmol/L must carry that, because a constant one primary
group could not identify at all is a different kind of object from
haloperidol's.

Vernaleken 2010's Table 1 does hold 16 rows of patient plasma concentration
against regional occupancy, which would be fittable. That table was read
through a page-summarising fetch rather than from retrieved source text, so it
is recorded here as unverified and is **not** counted as a REACHABLE_FITTABLE
source.

### Clozapine

Gründer G, Landvogt C, Vernaleken I, Buchholz HG, Ondracek J, Siessmeier T,
Härtter S, Schreckenberger M, Stoeter P, Hiemke C, Rösch F, Wong DF,
Bartenstein P. *Neuropsychopharmacology* 2006;31(5):1027–35. PMID 16237387.
[18F]fallypride, n=15 clozapine-treated inpatients, occupancy referenced to
7 normal volunteers.

Printed in the free PubMed abstract: ED50 950 ng/mL in putamen against
333 ng/mL in inferior temporal cortex, with Emax approaching 100% in both. Full
text is paywalled. Unpaywall reports this DOI as bronze OA, but
`nature.com/articles/1300931.pdf` returns the HTML landing page, so the OA flag
is wrong in practice and the abstract is all that is readable. Not fittable.

Two things make this the most caveat-heavy value in the set. The target is
**D2/D3** via fallypride rather than D2-selective. And the putamen, caudate and
cortex values span nearly 3-fold, so "the striatal EC50" is region-sensitive
here in a way it is not for the raclopride studies. The region has to be pinned
wherever this number is stored. The caudate value of 582 ng/mL circulates in
the literature but was readable only in Hart 2022 and a 2025 secondary paper,
not in Gründer's own text.

This value does not soften the clozapine anomaly P22.4 is built around. It
sharpens it: a striatal ED50 of 950 ng/mL sits well above clozapine's accepted
clinical range of 350 to 600 ng/mL, which is the point.

### Amisulpride, two sources that are not interchangeable

Amisulpride is the one drug where the printed value and the fittable data come
from different populations, and they must not be merged.

**Printed, elderly Alzheimer's disease.** Reeves S, Bertrand J, McLachlan E,
D'Antonio F, Brownings S, Nair A, Greaves S, Smith A, Dunn J, Marsden P,
Kessler R, Uchida H, Taylor D, Howard R. *J Clin Psychiatry* 2017;78(7):
e844–e851, PMID 28858444, with the same parameters in *Brain*
2017;140(4):1117–27, PMID 28334978. [18F]fallypride, oral 25 to 75 mg/day at
steady state. Printed: "Imax was estimated as 84.3% for the caudate (IC50
19.1 ng/ml…), 98.7% for the putamen (IC50 61.3 ng/ml…)", read first-hand from
the UCL Discovery accepted manuscript,
`https://discovery.ucl.ac.uk/id/eprint/1561234/1/Manuscript_J16-M11216_R1_101216.pdf`.
The basis is **total whole blood**, not plasma. The putamen estimate carries
RSE 70% and is poorly identified. The authors themselves attribute the
higher-than-expected occupancy per unit concentration to age- and
disease-related blood-brain-barrier changes, and note the cohort ran below the
100 to 319 ng/mL range used in schizophrenia. **This value is
population-specific and must not be used as a generic amisulpride EC50.** Doing
so would be the substitution rule 2 forbids, one population standing in for
another.

**Fittable, schizophrenia.** la Fougère C, Meisenzahl E, Schmitt G, Stauss J,
Frodl T, Tatsch K, Hahn K, Möller HJ, Dresel S. *J Nucl Med* 2005;46(6):
1028–33, PMID 15937316, free at
`https://jnm.snmjournals.org/content/jnumed/46/6/1028.full.pdf`. 123I-IBZM
SPECT, n=29 patients, oral amisulpride 50 to 1200 mg/day at steady state,
trough plasma drawn 12 to 14 h after the last dose. **It prints no EC50**, only
a correlation, r = 0.65. But Table 1 gives per-patient rows of dose, plasma
concentration in µg/L and striatal D2 occupancy, of which **28 are usable**
(patient 7 has no plasma value), spanning 27 to 910 ng/mL and 41.9% to 100%
occupancy. That is a fittable set in the schizophrenia population.

Three caveats on the fittable route: occupancy is a semiquantitative
`(STR - BKG)/BKG` ratio rather than a compartmental BP_ND, the control group is
not age-matched across the patient range, and one patient sits at exactly
100.0% occupancy, which will distort an unconstrained Emax fit.

No printed schizophrenia-population amisulpride EC50 was reachable free.
Vernaleken 2004 (PMID 15683553) is the paper most likely to contain one. It is
indexed as bronze OA but returns HTTP 403 at every route, and its abstract
prints only the 60% to 80% occupancy concentration band rather than a 50%
value.

Amisulpride is already excluded from the P22.2 label-dose test because it has
no US antipsychotic approval (P20.6). It remains in scope for P22.3.

### Paliperidone, the one that is not reachable

Arakawa R, Ito H, Takano A, Takahashi H, Morimoto T, Sassa T, Ohta K, Kato M,
Okubo Y, Suhara T. *Psychopharmacology (Berl)* 2008;197(2):229–35, PMID
18058087. This is the right paper: [11C]raclopride and [11C]FLB 457, n=13
patients with schizophrenia, oral paliperidone ER at steady state, and it does
report a striatal ED50 of 6.65 ng/mL as parent paliperidone.

It is not readable. The free PubMed abstract prints the ED50 **only in mg/day**
(2.38 striatum, 2.84 temporal cortex) with no concentration unit, so it fails
the printed criterion on the abstract alone. Full text is paywalled at
Springer. Unpaywall, OpenAlex and Semantic Scholar all report this DOI as green
OA pointing at `https://repo.qst.go.jp/records/45147`. That record is
metadata-only, with `access_right c_14cb` and zero file objects in both its
JSON export and its OAI-PMH record, so **the green-OA flag on this DOI is a
false positive**.

The 6.65 ng/mL figure was readable only in a Janssen medical-information page
and, as a 4.4 to 6.4 ng/mL range, in US patent 9439906B2 citing two Janssen PET
studies (SWE-1 and SIV-101) published as conference abstracts and not indexed
in PubMed. Both are secondary. Under rule 1 this cannot be written into the
model as a sourced primary value. It can only be recorded as an attributed
secondary reading, which is what this document does.

Whether a per-subject table exists behind Arakawa 2008 is **unknown** rather
than absent, because the full text could not be inspected to find out.

Do not substitute a risperidone active-moiety EC50 for paliperidone. The
active-moiety values (Remington, Gefvert, Uchida) sum parent risperidone and
9-OH-risperidone and are roughly an order of magnitude larger than a
paliperidone-only constant.

## The pooled table does not exist in the form the brief assumed

The phase brief treats Uchida 2011 (PMID 21508857) as the per-drug ED50 table
Phase 21 needs, and instructs that it be treated as unreachable. Both halves
were checked. The brief is right about the paywall and wrong about the
contents, and the second point changes the phase.

**Unreachable, confirmed.** Europe PMC reports `pmcid: null`, `inEPMC: N`,
`inPMC: N`, subscription required. Unpaywall reports `is_oa: false`,
`oa_status: "closed"`, `oa_locations: []`. OpenAlex and Semantic Scholar agree
(`CLOSED`, no repository full text). The PMC ID converter returns "Identifier
not found in PMC". journals.lww.com returns 403, and the Ovid landing page
serves the abstract only, containing zero occurrences of "ED50" or "ng/mL".
CORE holds a Crossref metadata stub with an empty download URL, BASE denies
access by IP, and the Toronto and Keio institutional repositories hold nothing.
The abstract prints prediction errors in percentage-occupancy units and no
concentration-unit ED50 anywhere. Every check in the brief reproduced.

**But the paper covers five drugs, not the eight this phase wants.** Its own
abstract states the inclusion rule, antipsychotics studied in 20 or more
subjects, and names the five that met it: risperidone (n=98), clozapine (n=75),
olanzapine (n=42), haloperidol (n=35), ziprasidone (n=31). There is no
quetiapine, aripiprazole, amisulpride or paliperidone ED50 in Uchida 2011.
Buying access would not have produced the table Phase 21 was designed around.
The per-drug primary route this task took was the only route that could ever
have worked.

That is worth stating plainly because it inverts the brief's risk model. The
assumption at risk was never "can we get Uchida". It was "does a pooled table
of this shape exist at all", and it does not.

For completeness, Uchida's five values are recoverable from open-access papers
that quote them with verified attribution to PMID 21508857. PMC3960457 and
PMC3627781 give risperidone active moiety 4.9 ng/mL with a=88.0%, olanzapine
7.1 with a=90.7% and ziprasidone 32.9 with a=88.2%. PMC9020768 (Hart 2022)
gives haloperidol 0.32 and 0.70, olanzapine 7 and 10, risperidone 5 and 8, and
clozapine 105 and 483 ng/mL for the unconstrained and Emax-constrained models
respectively. PMC12488982 (Spangemacher 2025, with Uchida as co-author) prints
a five-column table: aripiprazole 10, clozapine 483, haloperidol 0.32,
olanzapine 7, risperidone 5.

**These are secondary readings and are recorded as such.** Two hazards go with
them. The literature circulates both parameterisations without always labelling
which, and Spangemacher's single row mixes them, pairing the constrained
clozapine 483 with the unconstrained values for the other four. At least one
open-access source (PMC4681559) pairs the unconstrained Emax of 88.0% with the
constrained ED50 of 8.2 ng/mL, which is internally inconsistent. This
repository should prefer the primary per-drug values above.

Note also that Uchida's ziprasidone 32.9 ng/mL and Suzuki's primary 84 ng/mL
differ by 2.6-fold, and Uchida's clozapine 105 or 483 ng/mL against Gründer's
primary 950 ng/mL by 2- to 9-fold. Pooled and primary do not agree.

## An independent free pooled source, recorded but not adopted

de Greef R, Maloney A, Olsson-Gisleskog P, Schoemaker J, Panagides J. *AAPS J*
2011;13(1):121–30, PMID 21184291, PMC3032087, free to read. Table III prints
Emax and EC50 in ng/mL with standard errors: asenapine 101.8% and 0.528
(SE 0.063), haloperidol 92.0% and 0.532 (0.085), risperidone 91.2% and 4.43
(0.75), ziprasidone 98.0% and 15.4 (5.1), olanzapine 87.5% and 5.29 (1.22).

Two properties make this the best available cross-check and a poor ingestion
source. It is an independent fit rather than a re-quotation of Uchida, so
comparing the two is informative: de Greef's haloperidol 0.532 against Uchida's
0.32, olanzapine 5.29 against 7, risperidone 4.43 against 5. But its
"Relative PET EC50" column is corrected for molecular weight and plasma protein
binding, which makes those relative values a different quantity from the
total-plasma constants above and **not substitutable for them**. Recorded here
as an independent cross-check that exists rather than a source to ingest.

## What was rejected, and why

Recorded so the next task does not re-tread them.

- **Wrong species.** Muly 2012 (PMID 22214649), Suhara 2002 (PMID 11862380) and
  Nyberg 2002 (PMID 12057028) are macaque or cynomolgus studies. Muly is free
  in PMC and reports an attractive 40 to 80 ng/mL active-drug target for
  paliperidone. It is a monkey, and rule 2 forbids it. The EMA Sycrest EPAR
  likewise prints "Plasma levels of … 1.24 ng/ml were necessary for 50%
  receptor occupancy of the … D2" for asenapine, but that sentence sits in the
  non-clinical section and is rat.
- **Wrong drug.** Bressan 2003 (PMID 12544369) is cited in the phase context as
  amisulpride and is about risperidone. A flupentixol paper (PMID 17111172)
  prints "ED50 of about 0.7 ng/ml serum" for flupentixol, not for any drug in
  this set.
- **Wrong endpoint.** The EMA Okedi EPAR prints an EC50 "fixed to a value from
  literature (4.6 ng/mL)" for risperidone ISM. That is a **PANSS efficacy**
  EC50, not a D2 occupancy EC50, and the two are not interchangeable despite
  sharing units and an order of magnitude. This is the most dangerous near-miss
  found in the search.
- **Wrong region.** Arakawa 2010 (PMID 19851803) prints a plasma ED50 of
  10.5 ng/mL for olanzapine in the free abstract, but the region is temporal
  cortex via [11C]FLB 457 rather than striatum. Recorded as corroborating only.
  Takano 2006 for clozapine is extrastriatal, n=2, and re-uses a previously
  determined ED50 rather than fitting one.
- **Dose units, not concentration.** Martinot 1996 for amisulpride, a 2017
  *Nord J Psychiatry* olanzapine model, and Horowitz 2021 (PMC8266572) all
  report ED50 in mg/day. The chain needs a concentration.
- **Occupancy estimated, not measured.** Ikai 2016 (PMID 27631410) derives
  occupancy from plasma concentration through an EC50 rather than measuring it,
  which is circular for this purpose.
- **Two search summaries produced numbers that failed verification.** One
  attributed Gründer's clozapine 950 and 333 ng/mL to quetiapine. One produced
  Uchida ED50 values as though quoting the paper directly, when the paper was
  never retrieved. Both were caught by requiring that every value be re-read
  from retrieved text in this session before being written down. The Xiberas
  2001 BJP table was rejected on the same principle: its printed unit header is
  `mg/l`, which is wrong by a factor of 1000 given the values, and a source
  whose units are internally impossible cannot be transcribed under rule 1.
- **Not retrieved, so not characterised.** Kaar 2020 (*Neuropharmacology*, DOI
  10.1016/j.neuropharm.2019.107704) is the review most likely to hold a
  consolidated table. Unpaywall reports `is_oa: false` with zero OA locations
  and the KCL Pure record has no file, so its contents are unknown rather than
  negative. The same applies to the AGNP TDM guidelines (the Göttingen
  repository serves a bot interstitial) and to Pilla Reddy 2015 (Springer bot
  challenge).

## Routes that worked, and routes that lie

Recorded because the next task in this phase will hit the same walls.

Working: PubMed `efetch` for abstracts, which carried the printed value for
risperidone, clozapine, ziprasidone and aripiprazole. NCBI `efetch db=pmc`,
which returned full JATS for ziprasidone where Europe PMC's `fullTextXML`
returned 404 for the same PMCID. Europe PMC `fullTextXML`, which carried
quetiapine. The PMC website with a browser user-agent, which served Kim 2012's
methods where `efetch` returned a 10 kB stub. Institutional repositories, with
the University of Malta carrying both risperidone and olanzapine and UCL
Discovery carrying amisulpride. And the *J Nucl Med* open archive.

Unreliable: **the OA flags**. Unpaywall, OpenAlex and Semantic Scholar reported
open access for Gründer 2006 (bronze, nature.com serves the landing page),
Arakawa 2008 (green, the QST repository record has no file), Kodaka 2011
(green, same), Takano 2004 (same), Kessler 2005 (bronze, same landing page) and
Vernaleken 2004 (bronze, 403 at every attempt). Six false positives. An OA flag
is a hypothesis; only a retrieved body is evidence.

Blocked to non-browser clients: `academic.oup.com`, `psychiatryonline.org`,
`journals.lww.com`, `jamanetwork.com`, `onlinelibrary.wiley.com`,
`journals.sagepub.com` and `link.springer.com`, all 403 or bot challenge.
`accessdata.fda.gov` and `pmc.ncbi.nlm.nih.gov` both rate-limit aggressively;
`www.ncbi.nlm.nih.gov/pmc/articles/...` is a working alternate host for the
latter.

The FDA route was probed directly and holds population PK rather than PET
occupancy, which is exactly the division of labour P21.3 assumes. The Invega
Sustenna NDA 022264 clinical pharmacology review is text-searchable across
306,000 characters and contains zero hits for "EC50" or "receptor occupancy",
its only "D2" being a zero-order input duration parameter. Invega Trinza NDA
207946 cross-references the oral submission rather than restating occupancy.
Abilify Maintena NDA 202971 and Aristada NDA 207533 have no occupancy content.
Caplyta NDA 209500 prints in-vitro antagonist IC50 values (D2L 32 nM, D2S
70 nM in human CHO cells) and qualitative occupancy targets, but no plasma
EC50. The 2006 Invega NDA 021999 review PDFs are scanned images with no text
layer.

## What this task does and does not establish

It establishes that the load-bearing assumption of Phase 21 holds. Per-drug
plasma EC50 values for striatal D2 occupancy are obtainable from primary human
sources for eight of the nine candidate drugs, without paying for anything and
without touching a pirate mirror.

It establishes nothing about the model. No prediction was made, no comparison
against a measured number was run, and no code was written. This document earns
its ledger row in the untested and searched-and-empty tables rather than the
validated one.

It also leaves four things the ingestion task must handle rather than inherit
silently.

1. **The concentration bases are not the same quantity.** Risperidone is active
   moiety, quetiapine is offered as either, ziprasidone is serum, amisulpride
   is whole blood, and de Greef's independent table is protein-binding
   corrected. A registry with one `concentration_ng_per_ml` field and no basis
   field would erase this.
2. **Emax is inconsistent across the set.** It is fixed at 100% for ziprasidone
   and quetiapine, defined but never printed for olanzapine, freely fitted at
   87.5% to 101.8% by de Greef, and unidentifiable for quetiapine in a second
   primary study. These constants are not all the same kind of estimate.
3. **The primary-to-primary spread within a single drug is 2- to 3-fold** for
   ziprasidone, clozapine and risperidone. That is the same order as the tracer
   spread the ledger already records as a floor larger than every held-out
   error. Storing one number per drug hides it.
4. **The populations differ.** Healthy volunteers for quetiapine and
   aripiprazole, patients for the rest, and elderly Alzheimer's patients for the
   only printed amisulpride value.

The two drugs with fittable per-subject tables, risperidone (Remington Table 1,
18 pairs) and amisulpride (la Fougère Table 1, 28 pairs), should be refitted
in-repo rather than transcribed, on the P19.4 precedent. A transcribed constant
cannot be checked against its own data.
