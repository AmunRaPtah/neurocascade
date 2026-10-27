# P21.4: active metabolites that also occupy D2

The brief for this task said that risperidone is cleared to 9-hydroxyrisperidone,
which is paliperidone and is itself a marketed D2 antagonist, that aripiprazole
is cleared to dehydro-aripiprazole, also D2-active, and that a chain mapping only
parent concentration to occupancy will understate both silently. It asked for
either a summed-occupancy model with sourced parameters for each component, or an
explicit refusal with a recorded reason.

Both refusals were the outcome, and a third drug was cleared. The reasoning
behind each is below, together with the arithmetic that decides which of the
three answers a pair gets.

## The load-bearing assumption was checked first, and it is haloperidol's

The brief names risperidone and aripiprazole. Neither is in the chain. The drug
in the chain is haloperidol, and haloperidol has an active metabolite too:
reduced haloperidol, formed by ketone reduction and back-oxidisable to the
parent. If reduced haloperidol contributed materially to D2 occupancy, then V1,
the only validated claim this repository has, would be measuring something other
than what it says it measures, and both refusals below would be beside the point.

So that is what was checked first.

The US labels are no help at all, and the reason is the P20.6 finding showing up
again. The current generic oral haloperidol label (ANDA216918) has a CLINICAL
PHARMACOLOGY section that is 87 characters long in full: "The precise mechanism
of action has not been clearly established." Four current haloperidol decanoate
ANDA labels (219168, 214507, 205241, 074893) contain no occurrence of
"metabolit" or "reduced haloperidol" anywhere. All of them are generics'
copies of a withdrawn pre-modern innovator label, which is why the haloperidol
row in this module is sourced from the primary literature while the other two are
sourced from labels.

The primary literature gives one half of the answer cleanly and the other half
not at all.

- **Exposure, human, oral.** Uematsu 1992 (J Pharm Sci 81(10):1008-11, PMID
  1432610) measured ten Japanese patients on fixed oral maintenance doses of
  4 to 30 mg/day for more than four months: "The ratio of the area under the
  plasma concentration-time curve (AUC) of RHL for 1 day to that of HL also
  ranged widely from 0.39 to 1.99 (1.04 +/- 0.48, mean +/- standard deviation)."
  So the metabolite circulates at roughly the parent's own exposure.
- **Affinity, rat only.** Bowen 1990 (Eur J Pharmacol 177(3):111-8, PMID
  2155804) reports in rat striatal membranes that "reduced haloperidol bound to
  dopamine receptors with 85-fold lower affinity compared to haloperidol
  (KI = 239 nM)". Jaen 1993 (J Med Chem 36(24):3929-36, PMID 7902870) corroborates
  the direction for both separated enantiomers, "greatly decreased affinity for
  D2 and D3 receptors", in the 100 to 200 nM range.

No human-tissue D2 affinity for reduced haloperidol exists. Two targeted PubMed
searches returned Jaen 1993 alone and nothing else, and Jaen is a
medicinal-chemistry binding panel rather than human tissue. This is recorded as
searched-and-empty in the ledger.

Rule 2 forbids substituting a species. The rat 85-fold is therefore never used as
a human number; it is stored with `affinity_species="rat"` and everything derived
from it is reported as conditional on the ratio transferring. What it supports is
a negative conclusion, that the effect is too small to matter, which is the
direction in which a cross-species assumption is least dangerous, and the record
says so on its face.

Conditional on that transfer, the metabolite carries a drive ratio of
1.04 / 85 = 0.0122, an apparent-EC50 shift of 1.2%, and a worst-case occupancy
error over all doses of **0.0030**. The chain's own held-out mean absolute error
on V1 is 0.0351, so the metabolite sits 11.5-fold below the error the chain
already has. At the largest exposure ratio Uematsu observed in any single
patient, 1.99, the worst case is still only 0.0058. The bound survives the whole
observed range before the question of regimen is even asked, which matters
because Uematsu's ratio is oral and the repository's EC50 was fitted on depot
intramuscular steady state, where the ratio has not been measured.

Haloperidol is therefore dispositioned `bounded` rather than waved through, and
the bound rides on every prediction the chain makes.

## Why the failure is silent, and what actually decides its size

Two ligands competing for one receptor give

    occupancy = (C_p/K_p + C_m/K_m) / (1 + C_p/K_p + C_m/K_m)

If the metabolite tracks the parent at a fixed ratio `r = C_m / C_p`, substitute
and the whole thing collapses to

    occupancy = C_p (1 + d) / K_p / (1 + C_p (1 + d) / K_p),   d = r K_p / K_m

which is *still a rectangular hyperbola in the parent concentration alone*, with
apparent constant `K_p / (1 + d)`. That is why the error is silent. A parent-only
fit to two-ligand data does not fail, does not look wrong, and produces no
diagnostic. It returns a smaller number and relabels the metabolite's
contribution as the parent's potency.

Three things follow, and they are the entire decision procedure.

1. **Inside the regimen it was fitted in, a parent-only EC50 is not wrong.** It
   has already absorbed the metabolite. Adding a metabolite term on top of such
   a constant would count the metabolite twice, which is the opposite error and
   equally silent.
2. **Across a change of regimen it is wrong by however much `r` moved.** Whether
   `r` moves is an empirical question about two specific regimens, and the
   evidence for it is the two species' half-lives and Tmax values.
3. **The size is set by `d`, not by `r`.** A metabolite at equal exposure and
   85-fold weaker affinity contributes 1.2% of the drive. A metabolite at 40% of
   exposure and equipotent contributes 40%. Exposure ratio alone decides nothing,
   which is why a registry keyed on exposure ratio would have been useless.

`ActiveMetabolite.max_occupancy_error()` makes point 3 exact. Writing `g = 1 + d`
and `x = C_p / K_p`, the occupancy an ignored metabolite costs is
`x d / ((1 + g x)(1 + x))`, maximised at `x = 1/sqrt(g)` where it equals
`d / (1 + sqrt(g))**2`. The tests check that closed form against a brute-force
maximisation on a log-concentration grid rather than against the function's own
output, at six values of `d` spanning four orders of magnitude.

That closed form is what makes this a gate rather than a flag. Without it, every
active metabolite is a worry; with it, a metabolite either moves the prediction
by more than the chain's existing error or it does not.

## Aripiprazole: refused on a regimen mismatch

The ABILIFY label (NDA021436, openFDA `effective_time` 20250129) says:

> ABILIFY activity is presumably primarily due to the parent drug, aripiprazole,
> and to a lesser extent, to its major metabolite, dehydro-aripiprazole, which
> has been shown to have affinities for D 2 receptors similar to the parent drug
> and represents 40% of the parent drug exposure in plasma.

and, separately and more precisely:

> At steady-state, dehydro-aripiprazole, the active metabolite, represents about
> 40% of aripiprazole AUC in plasma.

The only in-vivo human aripiprazole EC50 P21.2 could reach is Kim 2012's
parent-only 11.1 ng/mL (95% CI 10.1 to 12.1), fitted after a **single oral dose**
in healthy young men. The 40% is explicitly **steady state**. Those are different
regimens, and the label itself supplies the evidence that the ratio differs
between them:

> The mean elimination half-lives are about 75 hours and 94 hours for
> aripiprazole and dehydro-aripiprazole, respectively.

> ...delayed T max by 3 hours for aripiprazole and 12 hours for
> dehydro-aripiprazole.

A metabolite with a longer half-life and a later Tmax accumulates to a larger
share of the moiety at steady state than it holds after one dose. So Kim's
constant has absorbed a smaller metabolite contribution than a steady-state
prediction needs, and carrying it into a steady-state chain misprices the drug by
the ratio of the two `(1 + d)` terms, neither of which is known.

The correction is not available. There is no published in-vivo human EC50 for
dehydro-aripiprazole alone, so the two-ligand form cannot be reconstructed and
the parent-only constant cannot be un-absorbed. The pair is refused.

A second parent-only constant does exist, and finding it did not clear the drug.
Gründer 2008 (*Am J Psychiatry* 165:988-95, [18F]fallypride, 16 patients at
steady state on 5 to 30 mg/day) reports EC50 values of 10 ng/mL in putamen and
9 in caudate, reachable second-hand through Hart, Schmitz & Gründer 2022 and
corroborated by Spangemacher 2025. A second parent-only constant is still a
parent-only constant: it has absorbed whatever metabolite was present in *its*
subjects, so it relocates the regimen mismatch rather than removing it.

What Gründer supplies that nothing else does is a **moiety-basis** constant for
the same subjects, 20 ng/mL in putamen and 18 in caudate. Two bases are not two
components and do not build the summed form. They do measure something else, and
this is the one piece of new method in P21.4.

### Two EC50 bases from one study measure the exposure ratio, and need no affinities

At half occupancy the two-ligand condition is `C_p/K_p + C_m/K_m = 1`. The
parent-basis EC50 is the parent concentration at that point, `P* = C_p*`; the
moiety-basis EC50 is the sum at the *same* point, `M* = C_p* + C_m*`. If the
metabolite tracks the parent at ratio `r`, then `M* = C_p*(1 + r)` and

    M* / P* = 1 + r

exactly. Both affinities cancel. They decide *where* half occupancy falls, and
the identity divides two descriptions of the same point. That matters here
because it means a repository which refuses to substitute an in-vitro Ki for an
in-vivo potency can still extract an exposure measurement from two published
constants. `exposure_ratio_from_ec50_bases()` is that identity; the test builds
occupancy from the two-ligand form with affinities swept over a 22-fold range in
both directions and recovers `r` to 1e-12, which is where a failure to cancel
would show.

Applied to Gründer's two regions, 20/10 and 18/9 both give **r = 1.0**. The
ABILIFY label says 0.40. Those do not reconcile. The reviews print the constants
as integers with no dispersion, so the check that matters is whether rounding
alone could close the gap: it cannot, because the rounding interval most
favourable to the label still bounds the ratio at 0.86 to 1.16 in putamen and
0.84 to 1.18 in caudate. The same review is internally inconsistent about it too,
saying the metabolite "amounts to up to 40% of the parent concentrations" and
carries "usually 20-30%" of total occupancy. An occupancy share of 20 to 30% is
`d` of 0.25 to 0.43, while the EC50 pair printed two sentences later implies
`d = 1.0`, a 50% share.

So the exposure ratio for aripiprazole is not settled to better than 2.5-fold by
the reachable sources. That is a third reason for refusal rather than a bound: a
bound needs a ratio, and here the sources disagree about which ratio it is.

A fourth and independent reason to refuse is the size. If the label's
"affinities... similar to the parent drug" were taken at face value, `d = 0.40`
and the worst-case occupancy error from ignoring the metabolite is **0.084**,
2.4-fold above the chain's held-out MAE. That figure is an illustration of the
scale of the problem and is deliberately not a stored parameter: the label
prints aripiprazole's own D2 Ki (0.34 nM) and no Ki at all for
dehydro-aripiprazole, so equipotency here is an unnumbered assertion.
`affinity_ratio` stays `None`, the assertion lives in the record's `quotes` where
it cannot be arithmetically consumed, and `drive_ratio()` raises.

## Risperidone: refused because the parent-only path is ill-posed

The RISPERDAL label (NDA020272/NDA020588/NDA021444, `effective_time` 20260528)
does not hedge:

> The clinical effect from risperidone results from the combined concentrations
> of risperidone and its major metabolite, 9-hydroxyrisperidone (paliperidone).

> The main metabolite, 9-hydroxyrisperidone, has similar pharmacological
> activity as risperidone.

P21.2 had already found that every reachable in-vivo risperidone value is on the
summed active moiety and that no parent-only risperidone EC50 exists anywhere.
So on the parent axis the constant is simply absent, which alone would force a
refusal.

The stronger reason is the next sentence on the label:

> Although extensive metabolizers have lower risperidone and higher
> 9-hydroxyrisperidone concentrations than poor metabolizers, the
> pharmacokinetics of risperidone and 9-hydroxyrisperidone combined, after single
> and multiple doses, are similar in extensive and poor metabolizers.

The split between parent and metabolite is CYP2D6-phenotype dependent; the sum is
not. The recorded half-lives make the same point: 3 h parent and 21 h metabolite
in extensive metabolisers, 20 h and 30 h in poor metabolisers, with the combined
half-life near 20 h in both. So parent risperidone concentration is not a
well-defined function of dose across a population. The same dose produces
systematically different parent concentrations in the two phenotypes while
producing the same moiety.

A parent-only path for risperidone is therefore ill-posed rather than merely
unsourced, and no future search closes it. That distinction is kept in the
registry, because aripiprazole becomes usable the day a dehydro-aripiprazole
EC50 is published and risperidone does not. The tests assert the two refusals
differ in exactly this way, so that a later refactor cannot collapse them into
one rule.

### The refusal is priced, not argued

Risperidone is the one refused pair where both ratios turned out to be
obtainable, so the cost of the refusal is a number rather than a worry.

- **Exposure, human, oral.** Lenk et al. 2022 (*Clin Pharmacol Ther* 111(5):1165-74,
  PMID 35253216) measured the metabolite:parent serum ratio in a therapeutic
  drug monitoring cohort stratified by CYP2D6 phenotype and NFIB genotype, ten
  subgroups spanning **0.25 to 33.11**, a 130-fold range across one table. The
  row used here is CYP2D6 normal metabolisers with NFIB TT: **9.93 (95% CI 8.55
  to 11.54), 243 patients**. The RISPERDAL label prints no ratio at all.
- **Affinity, human, same assay.** Schotte 1995 (*Jpn J Pharmacol* 69(4):399-412,
  Table 4) measured both ligands against cloned human D2L in CHO cells with
  [3H]spiperone: risperidone Ki 5.9 nM, 9-OH-risperidone Ki 4.8 nM. That is a
  ratio of two same-assay constants, which is what sets relative occupancy drive;
  it is not an in-vitro constant standing in for an in-vivo one, and the
  haloperidol record already rests on the same kind of ratio.

Together: `d = 9.93 × (5.9/4.8) = 12.2`. The metabolite carries about twelve
times the parent's occupancy drive, and the worst-case occupancy error from
ignoring it is **0.57**, sixteen times the chain's own held-out MAE of 0.0351.
The same closed form that clears haloperidol at 0.0030 condemns risperidone at
0.57. That is the quantitative form of what the brief called understating the
drug silently.

The conclusion does not rest on the weaker of the two inputs. Stressed at an
affinity ratio ten times more favourable to the parent than the measured one,
`d` falls to 1.22 and the error is still 0.197, nearly six times the MAE. It is
the exposure ratio near 10 that decides this drug. Even at the bottom of Lenk et al.'s
130-fold range, the poor-metaboliser ratio of 0.25, the worst-case error is 0.067,
still about twice the MAE. No CYP2D6 phenotype makes the parent-only path safe;
the phenotypes differ in how badly it fails, which is the phenotype dependence
again, stated in occupancy units.

The label also records that risperidone is about 90% protein-bound and
9-hydroxyrisperidone 77%. Equipotency on total concentrations and equipotency on
free concentrations are therefore different claims, and nothing here assumes
either.

## The one comparison available, and it misses

Rule 9 wants a prediction against a number nobody here chose. The active-moiety
problem offers exactly one, and it exists only because of the coincidence the
brief pointed at: risperidone's metabolite is itself an approved drug with its
own PET study.

Arakawa 2008 measured D2 occupancy in subjects taking paliperidone ER, who
therefore had no parent risperidone in them at all, and reports a striatal ED50 of
6.65 ng/mL for paliperidone alone. Remington 2006 measured the risperidone
**active moiety** and reports an ED50 of 11.06 ng/mL (95% CI 9.15 to 12.96, Emax
fixed at 100%) on the long-acting injectable, and 6.08 ng/mL on oral.

The label says the two components are equipotent. If that is true, then for a
moiety `M = C_r + C_p` with the parent holding fraction `w` of the mass, competitive
binding puts half-occupancy at the weighted harmonic mean

    M = 1 / (w/K_r + (1 - w)/K_p)

with `K_r` obtained from `K_p` by the molecular-weight ratio, because equipotency
is a claim about molar affinity and the moiety assay reports a sum of masses.
Risperidone is 410.49 g/mol and paliperidone 426.45, so `K_r` = 6.40 ng/mL and the
predicted moiety ED50 is **6.52 ng/mL**, moving only between 6.40 and 6.65 across
every possible composition. The prediction does not need the moiety composition,
which is fortunate, because it is not known for either study population.

Measured, on the LAI: 11.06 ng/mL. **The prediction misses by 1.70-fold, in the
direction of predicting more potency than was observed.**

That is the result and it is not tuned. Four candidate explanations are attached
to the check itself rather than argued here, because each of them could account
for the whole gap on its own:

1. **Provenance.** Arakawa 2008 is reachable second-hand only. P21.2 read
   6.65 ng/mL from the sponsor's Invega medical-information page, which cites
   Arakawa as its source; the primary is paywalled and its free abstract prints
   ED50 in dose units, not concentration. This is the weakest provenance class in
   the repository. The same sponsor's patent US 9,439,906 B2 gives an apparent KD
   of 4.4 to 6.4 ng/mL from two unpublished healthy-volunteer PET studies, which
   spans the same region and would move the prediction down, not up.
2. **Emax convention.** Remington fixed Emax at 100%. Whether Arakawa did is not
   established. P21.2 measured that this single choice moves clozapine's constant
   4.6-fold and haloperidol's 2.2-fold, so a mismatch could produce a 1.70-fold
   gap by itself.
3. **Regimen.** Remington's 11.06 is LAI steady state. The same group's oral
   figure is 6.08 ng/mL, and the prediction of 6.52 sits within 7% of that. If the
   comparison is made against the oral value the check passes comfortably, which
   suggests a substantial part of the gap is depot pharmacokinetics rather than a
   failure of equipotency. It is recorded against the LAI value because that is
   the figure P21.2 transcribed as the drug's row.
4. **Basis.** Equipotency on total plasma against equipotency on free plasma are
   different claims given the 90% / 77% binding difference, and this check tests
   only the former.

The check is also not blind, and the ledger row says so. The Remington value was
already transcribed into this repository by P21.2 before P21.4 ran, so the
prediction was assembled with the answer visible. It is recorded on the same
footing as V4, which carries the same caveat for the same reason.

## Where the gate sits, and why the ordering is deliberate

`require_parent_only_path()` is called from `_require_receptor_record()` in
`dose_to_occupancy.py`, **before** the check that the record's ligand is
haloperidol and before the existing summed-moiety analyte check.

Both of the later checks would also reject a risperidone record, so the ordering
looks cosmetic and is not. The three messages recommend three different repairs:

- The ligand check says "this chain is haloperidol's", which invites a caller to
  generalise the chain. Doing that would reopen exactly the hole P21.4 exists to
  close, because the generalised chain would then accept a parent concentration
  for a drug whose parent concentration does not determine its occupancy.
- The summed-moiety check says the constant is on the wrong axis, which invites a
  caller to go and find a parent-only constant. For risperidone that constant
  does not exist, and following the advice would repeat a search P21.2 already
  ran to exhaustion.
- The metabolite check names the metabolite and the reason.

A test asserts the ordering by handing in a risperidone record explicitly flagged
`analyte_is_summed=False`, so that the pre-existing gate cannot be what rejects
it, and asserts the message names 9-hydroxyrisperidone.

Haloperidol's `bounded` record is attached to every prediction as
`OccupancyPrediction.metabolite_caveats` and printed by `summary()`. It is
deliberately **not** a `ChainLayer`. A layer's provenance is a claim about a
number the prediction consumed, and the chain consumes nothing from this record:
the parent-only EC50 already absorbs whatever reduced haloperidol contributed in
the depot regimen it was fitted in. Making it a layer with provenance
`illustrative` would additionally poison the whole prediction's provenance and
break V1's own validation for a metabolite that moves the answer by 0.003.

## What was not built

No summed-moiety occupancy model, for any drug. It would need an in-vivo human
EC50 for the parent and one for the metabolite, on one concentration basis with
one Emax convention. The published axes are: risperidone, moiety only;
aripiprazole, parent and moiety; haloperidol, parent only. **No registered pair
has a metabolite axis**, which is the one the summation needs. Aripiprazole's two
axes are two views of the same mixture, not the two components, so a summation
function would still be a container for data that does not exist, which rule 10
rules out. What the two axes are good for instead is the exposure ratio, and that
is implemented.

What is built instead is a registry that decides each case from sourced numbers:
the 0.40 AUC fraction against Gründer's derived 1.0, and the 75/94 h half-life
pair, for aripiprazole; Lenk et al.'s 9.93 exposure ratio against Schotte's same-assay
human Ki pair, plus the 3/21 h phenotype-dependent half-lives, for risperidone;
and the 1.04 exposure ratio against the 85-fold rat affinity ratio for
haloperidol. Those numbers are what produce the two refusals and the one
clearance; remove them and the module has nothing to say. Two of the three
dispositions are now decided by arithmetic on measured ratios rather than by
argument: 0.0030 clears, 0.57 refuses, and the threshold in both cases is the
chain's own held-out error rather than a number chosen here.
