# P19.4: the in-vivo D2 EC50 and the in-vitro Ki correction

The brief asked for two numbers to be stored separately rather than collapsed
into one affinity slot: a human in-vivo plasma EC50 for haloperidol striatal D2
occupancy, and a human in-vitro D2 Ki with its assay conditions. It also asked
for the ratio between them, and for an explicit statement of which one the
occupancy model consumes.

## Why the registry had to change shape first

`HumanReceptorParameter` carried a single `ki_nM` field. Nothing could be stored
in it honestly. An in-vitro Ki and an in-vivo EC50 share units and nothing else:

- The Ki comes out of a displacement assay and depends on what it was displacing.
  Seeman & Van Tol 1995 measured neuroleptic affinities at cloned human D2long
  and D2short against three radioligands and found the apparent Ki moving
  systematically with that choice, by more than tenfold across
  [3H]nemonapride, [3H]spiperone and [3H]raclopride. Writing down "the" in-vitro
  Ki without naming the radioligand records a number with an unstated tenfold
  ambiguity in it.
- The EC50 is the plasma concentration at which half the striatal receptors are
  occupied in a living person. It absorbs plasma protein binding, brain
  penetration, competition with endogenous dopamine, and the tracer used to
  measure the occupancy.

A second problem was that a Ki and an efficacy belong to a ligand at a receptor,
not to the receptor, and the record had no ligand field at all. It now does. The
D2 record names haloperidol; the five records still without a source leave
`ligand=None`, which states that no ligand has been chosen for them yet.

The occupancy path consumes the in-vivo EC50.
`HumanReceptorParameter.occupancy_constant_nM()` returns it and raises if it is
absent. Getting an in-vitro Ki out of that method requires passing a correction
factor **and** naming its source, so a Ki cannot reach the model wearing an
in-vivo label. `usable` was redefined to require the in-vivo EC50: a record
holding only a Ki stays unusable however well cited the Ki is.

## The in-vivo EC50: fitted here, not transcribed

Two published relations were already in the repository from P19.3, Kapur 1997 at
0.51 ng/ml and Fitzgerald 2000 at 0.40 ng/ml. Either could have been transcribed
into the D2 record in one line. Both were declined, for two reasons.

The first is circularity. P19.6 validates the finished chain against the three
Kapur occupancy ranges. If the chain's central constant were Kapur's own fitted
EC50, that validation would be checking a model against the model it came from.
Fitting in-repo leaves all three Kapur ranges and both published relations
untouched by the fit, so P19.6 has something to test against.

The second is reproducibility. `fit_plasma_ec50()` recomputes the constant from
the committed fixtures every time it is called, and a test asserts that the
recorded `HALOPERIDOL_D2_PLASMA_EC50_nM` still equals the recompute. A
transcribed number has no such guard; it is only as good as the transcription.

### What the fit rests on

Of the fifteen observations P19.3 registered, four carry both a central
occupancy and a **total** plasma concentration. All four are group means from the
two haloperidol decanoate depot studies:

| Study | Timepoint | Total plasma | Occupancy | n |
|---|---|---|---|---|
| Nyberg 1995 | week 1 | 4.6 nM | 0.73 | 8 |
| Nyberg 1995 | week 4 | 2.3 nM | 0.52 | 8 |
| Regenthal 1997 | week 1 | 7.3 nM | 0.75 | 8 |
| Regenthal 1997 | week 4 | 1.8 nM | 0.53 | 8 |

Four points, but only two independent patient groups: each study measured the
same eight patients twice. The interval below is reported with that caveat
attached in `EC50Fit.interval_caveat`, in words, so it cannot be quoted as if it
came from four independent observations.

Both studies are depot intramuscular steady state in patients. The chain being
built is for oral dosing. That mismatch is not repaired anywhere and is carried
forward to P19.7.

### The fit

Model: `occupancy = C / (C + EC50)`, a rectangular hyperbola with no baseline
term and no Emax term. Unweighted residuals on occupancy, because the sources
print group means without standard errors, so there is nothing to weight by.
Solved by bisection on the derivative of the residual sum of squares rather than
by an optimiser, so the result does not move with a scipy version. A test
confirms the returned value is the least-squares optimum by perturbing it.

| Quantity | Value |
|---|---|
| EC50 | **1.885 nM**, equivalently **0.708 ng/ml** |
| Asymptotic standard error | 0.176 nM |
| 95% interval, profile RSS with F(1, n-1) | 1.382 to 2.525 nM (0.519 to 0.949 ng/ml) |
| Residual RMSE in occupancy units | 0.036 |
| Nyberg 1995 alone | 1.940 nM |
| Regenthal 1997 alone | 1.817 nM |

The two studies were done by different groups, and their independent fits land
7% apart and bracket the pooled value. That replication is the reason a
four-point fit is worth recording at all, and it is asserted as a test rather
than only claimed here.

### How it compares to what is already published

| Source | Tracer | EC50, total plasma | Ratio, ours over theirs |
|---|---|---|---|
| Kapur 1997 | [11C]raclopride | 1.357 nM (0.51 ng/ml) | 1.39 |
| Fitzgerald 2000 | raclopride lineage | 1.064 nM (0.40 ng/ml) | 1.77 |
| **This repository** | not named in either source | **1.885 nM** | 1.00 |
| Videbæk 2001, converted | [123I]IBZM | 2.554 nM | 0.74 |

Both published constants fall **below** the lower bound of our interval, and the
IBZM-derived constant falls just above the upper bound. This is asserted as a
test, not smoothed over: no single hyperbola in this repository reproduces the
oral raclopride literature and the depot means at the same time.

The Videbæk row is the only converted entry. Its fit is on free plasma, 0.230 nM,
divided by the paper's own mean nonprotein-bound fraction of 9.0%. Total and free
observations are never pooled into one fit, because they differ here by about
elevenfold and a pooled fit would be fitting the protein binding rather than the
receptor. `ec50_landscape()` is the one place the conversion is applied, and it
labels that row as converted.

A separate check on the free-basis fit: it reproduces the mean of the four
per-subject Kd values Videbæk printed (0.16, 0.24, 0.19, 0.39 nM) to within 0.06
nM. That is a check on the transcription as much as on the model.

### What the spread means

The 1.39-fold gap between our depot fit and Kapur's oral fit is smaller than the
2 to 2.5-fold tracer spread P19.3 already recorded between raclopride-derived and
IBZM-derived constants. So the depot-versus-oral difference, whatever it is, is
not the largest term here. The tracer is, and neither depot study names one.
Nothing in this task narrows that; it sets the floor on what P19.6 can expect to
match.

## The in-vitro Ki

The search returned not one number but seven, and the spread between them is the
result. All seven are primary measurements at cloned human D2, all retrieved in
full text, and they run from 0.13 to 8.85 nM: a factor of 68. They are in
`neurocascade.human.d2_in_vitro`, lowest first, each with the conditions needed
to read it.

| Label | Ki, nM | Radioligand | Class | Isoform | Sodium | Temp |
|---|---|---|---|---|---|---|
| roberts2005 | 0.13 | [3H]NPA, 1 nM | agonist | D2S | none, NMDG substituted | 25 °C |
| sykes2017 | 0.33 | PPHT-red, 12.5 nM | fluorescent agonist | D2L | 138 mM + GppNHp | 37 °C |
| fan2020 | 0.46 | [3H]-N-methylspiperone, 0.8-1.0 nM | antagonist | D2L | none in recipe | room temp, not printed in °C |
| hall1997 | 2.9 | [3H]spiperone, conc. not printed | antagonist | D2S | 100 mM + 100 µM GTP | not printed |
| kaczor2023a | 6.56 ± 0.88 | [3H]spiperone, 0.6 nM | antagonist | D2S | not printed | not printed |
| kaczor2023b | 7.41 | [3H]spiperone, 0.6 nM | antagonist | D2S | not printed | not printed |
| bartuzi2022 | 8.85 | [3H]spiperone, 1.5 nM | antagonist | D2S | 120 mM | 25 °C |

Two patterns are worth naming, and one common explanation does not survive
contact with the table.

The spread tracks tracer class and guanine-nucleotide state. The three
[3H]spiperone filtration assays sit between 2.9 and 8.85 nM; the agonist-tracer
and [3H]-N-methylspiperone assays sit between 0.13 and 0.46 nM. This is the
effect Seeman & Van Tol 1995 reported across three radioligands, reproduced here
from sources that could be retrieved. That paper is paywalled and contributes no
number to this repository.

Sodium is the usual explanation and it does not carry the table by itself. The
highest sodium of any source, Sykes 2017 at 138 mM, reports one of the lowest
values, and two of the [3H]spiperone sources never print their buffer at all, so
the ion effect cannot be separated from what was printed. The `not_printed` tuple
on each record exists for this: a blank field means the paper did not say, not
that the condition was checked and found equal.

The second pattern is inside single papers. Kaczor 2023a reports a binding Ki of
6.56 nM and its own functional cAMP Kb of 0.11 nM; Kaczor 2023b reports 7.41 nM
against 0.29 nM. Same laboratory, same receptor, same paper, 20 to 60-fold apart.
The functional constants are registered separately in
`HALOPERIDOL_HUMAN_D2_FUNCTIONAL` and never enter the binding set, because a
ratio taken against the in-vivo EC50 lands somewhere different depending on which
kind of constant is the denominator.

### Which one the registry cites, and why

`REGISTRY_KI_LABEL` is `bartuzi2022`, 8.85 nM. It was chosen on completeness of
printed conditions and not on its value: it is the only source that prints
radioligand and its concentration, human isoform, sodium, temperature,
incubation time and the nonspecific-binding definition together. It is also the
highest of the seven, so the choice cannot be an attempt to make the in-vitro and
in-vivo numbers agree. A test asserts both facts.

### The ratio

`in_vitro_in_vivo_ratios()` returns the full set rather than one number, because
one number would be a choice of denominator worth 68-fold:

| Label | Ki, nM | EC50 / Ki |
|---|---|---|
| roberts2005 | 0.13 | 14.50 |
| sykes2017 | 0.33 | 5.71 |
| fan2020 | 0.46 | 4.10 |
| hall1997 | 2.9 | 0.65 |
| kaczor2023a | 6.56 | 0.287 |
| kaczor2023b | 7.41 | 0.254 |
| **bartuzi2022**, registered | **8.85** | **0.213** |

The range 0.21 to 14.5 **straddles unity**, and that is the argument against ever
substituting one quantity for the other. The in-vitro literature does not agree
on whether the in-vivo constant is larger or smaller than it is, so there is no
correction factor to apply, not even a badly determined one. No correction factor
is recorded anywhere in this repository. `occupancy_constant_nM()` returns the
in-vivo EC50 and will only return a scaled Ki if the caller supplies a factor
*and* names its source.

### What was not found

No primary human **tissue** measurement was reachable. Searches over human
post-mortem brain, caudate, striatum and pituitary returned reviews and
conference-abstract collections only. Every retrievable primary human D2
haloperidol affinity comes from a heterologous cell line, CHO-K1 or HEK293T. A
test asserts that this remains true of every registered record, so the note and
the data cannot drift apart.

Four sources were examined and excluded, each with its reason recorded in
`EXCLUDED_SOURCES`: de Witte 2018, whose binding assay is never stated to use a
human receptor; McCormick 2010, which is rat brain; Kroeze 2003, paywalled with a
false open-access signal, from which no number was taken; and the PDSP Ki
database, an aggregator rather than a primary source. Pirate mirrors are out of
scope for this repository, so paywalled means unavailable.

## Intrinsic efficacy: haloperidol is an inverse agonist

The efficacy search was not empty, and it changed what the record says.
Haloperidol is an inverse agonist at human D2, not a neutral antagonist, by two
independent functional readouts:

| Source | Readout | Efficacy | As printed |
|---|---|---|---|
| hall1997-cAMP | forskolin-stimulated cAMP, whole cells | **-0.95** | maximal effect 0.95 ± 0.03 vs 1 µM (+)-butaclamol, n = 3 |
| roberts2005-GTPgS | [35S]GTPγS binding, membranes | -0.83 | 83 ± 6 % of the maximal (+)-butaclamol response, n = 5 |

The registered value is -0.95, from the whole-cell cAMP assay, because that
readout is one step closer to the downstream effect the model represents and
because its authors report haloperidol's maximum as statistically
indistinguishable from their reference compound's. The other value is registered
alongside it and the -0.83 to -0.95 range is stated rather than averaged away.

The sign convention is asymmetric and is written down in `EFFICACY_SCALE`,
because it is easy to misread as a symmetric axis. Zero is a neutral antagonist.
Positive values are agonism referenced to the endogenous agonist. Negative values
are inverse agonism referenced to **(+)-butaclamol's maximal inverse response in
that assay**. So -0.95 means 95% of butaclamol's effect, not 95% as far below
basal as dopamine is above it. The two ends of the scale do not share a reference
and nothing converts between them.

Three limitations are carried on the records themselves:

- **The two quantitative sources are from the same laboratory** (Strange,
  University of Reading). They agree on direction and magnitude, but they are not
  independent in the way two laboratories would be.
- **Roberts 2005's 83% is condition-specific.** The authors omit sodium to
  increase constitutive receptor/G-protein coupling and state that inverse
  agonist signals were greater with no or low GDP. The number is a property of
  those conditions as much as of the drug.
- **Both systems use high heterologous receptor expression**, which is what makes
  constitutive activity measurable at all.

Independent support for the direction exists and carries no number. Burstein
2005, an unconnected laboratory, reports that virtually all the antipsychotics
tested were D2 and D3 inverse agonists. It is paywalled, its abstract does not
name the functional readout, and no efficacy value is printed, so it is recorded
as a qualitative statement in `INVERSE_AGONISM_ABSTRACT_ONLY_SUPPORT` and is
excluded from the efficacy set.

One search came back empty and is recorded as such: no primary β-arrestin
measurement of haloperidol's intrinsic efficacy at human D2 was retrievable
across thirty screened open-access records. Constitutive β-arrestin recruitment at
D2 is not routinely measurable, so this may be a real absence in the literature
rather than a retrieval failure. Only the non-retrieval is claimed.

## What the model consumes

The occupancy path consumes the **in-vivo EC50, 1.885 nM on a total-plasma
basis**. The in-vitro Ki of 8.85 nM is stored in the same record and reaches
nothing. `usable` requires the in-vivo EC50, so a record holding only a Ki stays
closed however well cited that Ki is, and a test constructs exactly such a record
to check it. The D2 record is now the only usable one of the six; the other five
still have neither an EC50 nor an efficacy and still refuse a rodent fallback.
