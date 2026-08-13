"""P5.7: the kappa and delta opioid receptors, and the end of mu selectivity.

Units, as everywhere else in this codebase: concentration in nM, time in
minutes, `kon` in nM^-1 min^-1, `koff` and `ke` in min^-1.

What is calibrated here and what is not
---------------------------------------

CALIBRATED (literature-sourced, cited per parameter): the affinity of all
seven ligands at ALL THREE opioid receptors, from one laboratory, one set of
assays and one preparation ([Toll1998]); the EFFICACY of all seven at all
three, from the same paper's [35S]GTPyS panel on the same cell membranes; a
measured association rate constant at kappa and another at delta; and the
`connection_modulation` gain of the kappa system, which like CB1's is a
measurement rather than an illustrative constant.

ILLUSTRATIVE: the network, which is
`scenarios.multi_region_mu_opioid_network()` reused unchanged; the delta
mechanism gain; and the pharmacokinetics of every ligand except fentanyl and
naloxone, which keep the calibrated PK Track A gave them.

The sentence in the brief that set the task
-------------------------------------------

"Fentanyl and naloxone are not mu-selective in the limit, and naloxone's
kappa and delta affinities matter for reversal modeling."

That is checkable rather than rhetorical, and the numbers say it more
sharply than the sentence does. In [Toll1998]'s human cloned-receptor panel:

    ligand         mu Ki    delta Ki   kappa Ki     mu-selectivity
    fentanyl        0.7      152.7       84.8       218x / 121x
    naloxone        1.4       67.5        2.5        48x /   1.8x

**Naloxone is 1.8-fold selective for mu over kappa.** Not 100-fold, not
10-fold: 1.8-fold, which is inside the spread of a single competition curve.
Naloxone is a mu-preferring but essentially non-selective opioid antagonist,
and any dose of it large enough to reverse a mu agonist has also blocked most
of the kappa receptor pool. `naloxone_mu_over_kappa_selectivity()` exists so
that this is a test rather than a paragraph.

Fentanyl's selectivity is real but finite, and it has a second half that a Ki
table alone hides: fentanyl is a PARTIAL AGONIST at kappa, 30 +/- 4% of
U69,593 in [Toll1998]'s GTPyS panel, and a near-full agonist at delta,
86 +/- 19% of DPDPE. So at a high enough concentration fentanyl does not
merely occupy kappa, it activates it, in the direction opposite to its own mu
effect. `test_opioid_scenario.py` measures how big that is in this model and
`results/opioid_cross_affinity.txt` records it.

Where the three receptors go, and why each is a citation
--------------------------------------------------------

No new circuit is drawn. `scenarios.multi_region_mu_opioid_network()` --
VTA_GABA -| VTA_DA -> NAc -> PFC, plus the unconnected PAG and LC blocks --
is reused character for character, so a kappa result here is comparable with
the mu results this codebase has had since Track A. This is the third P5
module to add no network of its own (after P5.5 and P5.6c) and the first to
place two new receptors on one existing one.

* **MU**: unchanged. `scenarios.mu_opioid_multi_region_coupling()` on
  VTA_GABA, PAG_Interneuron and LC, with P4.3's atlas-derived densities.

* **DELTA on `VTA_GABA`, via `coupling`, negative.** [Devine1993]
  microinjected DPDPE into the rat VTA and measured ventral striatal
  dopamine by microdialysis: dose-orderly INCREASES, antagonised by
  intra-VTA naltrindole and NOT by the mu antagonist CTOP, while DAMGO's
  increases were antagonised by CTOP and not by naltrindole. So the delta
  receptor that raises mesolimbic dopamine is in the VTA, is not the mu
  receptor, and acts through the same Gi-on-a-GABAergic-interneuron
  disinhibition motif. That is `VTA_GABA` with a negative gain, the same
  shape as mu on the same node.

* **KAPPA on the `VTA_DA -> NAc` CONNECTION, via `connection_modulation`,
  negative and MEASURED.** Three findings force this rather than a
  `coupling` entry on a node:

  1. [DiChiara1988] gave freely-moving rats systemic mu agonists (morphine,
     methadone, FENTANYL) and kappa agonists (U-50,488, bremazocine,
     tifluadom) and measured accumbal dopamine by dialysis: the mu agonists
     RAISED dopamine release, the kappa agonists LOWERED it. Two of the four
     drugs this module builds are in that experiment, so the opposite
     valence is not being imported from a related compound.
  2. [Devine1993] microinjected U-50,488H into the VTA and it "failed to
     alter ventral striatal DA concentrations at any dose tested", while
     "subsequent systemic injections significantly decreased" them. The
     kappa receptor that lowers accumbal dopamine is therefore NOT on the
     VTA cell bodies.
  3. [Svingos2001] found by immuno-electron microscopy that kappa-receptor
     labelling in the rat accumbens is mainly PRESYNAPTIC and that 51% of
     400 kappa-labelled presynaptic profiles also expressed the dopamine
     transporter, i.e. the receptor sits on the dopaminergic terminal
     itself.

  A receptor on a presynaptic terminal that reduces transmitter release at
  one identified synapse is exactly what `connection_modulation` represents
  and exactly what a single-node `coupling` entry cannot: this is P5.6b's
  CB1 argument, reappearing for a different receptor, and it is the reason
  `coupling_kappa()` returns an empty dict.

  [Margolis2003] is the one paper that complicates this, and it is not
  hidden: in VTA slices a selective kappa agonist (U69593, 1 uM) directly
  inhibits tyrosine-hydroxylase-positive neurons through a GIRK current,
  blocked by nor-BNI. So a somatodendritic kappa receptor on VTA dopamine
  neurons does exist and is functional in slice; what [Devine1993] shows is
  that it is not the one carrying the in vivo dopamine decrease. This module
  models the terminal receptor and does not model the somatodendritic one,
  and that is a choice between two measured mechanisms rather than a gap.

The kappa gain is a measurement, and so is "the antagonist alone is inert"
--------------------------------------------------------------------------

[Britt2008] recorded electrically evoked dopamine overflow in rat striatal
slices with fast electrochemistry and bath-applied a SATURATING
concentration (1 uM) of the selective kappa agonist BRL52537. Single-pulse
evoked dopamine overflow fell **to 28 +/- 5% of control**, in every
recording, throughout the dorsal and ventral striatum, and unlike the mu and
delta effects in the same paper it could not be overcome by high-frequency
burst stimulation. Nor-binaltorphimine (1 uM) "mostly restored the DA
signal", and -- the second measurement this module leans on -- "alone had no
direct effect on DA overflow".

So `KOR_MAX_SUPPRESSED_FRACTION = 0.72` and `KOR_CONNECTION_GAIN = -0.72`,
the same construction P5.6b used for CB1: the engine forms
`1 + gain * activation`, a saturating full agonist drives `activation` to
~1, and the modulated weight retains 28% of its strength. Two consequences,
both asserted by tests:

1. **A kappa agonist in this model cannot silence the mesolimbic
   projection.** It can remove at most 72% of it, because that is what was
   measured. The excitatory weight `W[NAc, VTA_DA] = +2.0` stays excitatory
   for every attainable activation; a gain below -1 would silently invert it
   and nothing in the engine forbids that.
2. **Nor-BNI alone changes nothing here, and for once that is a measured
   prediction rather than an artefact of the missing transmitter tone.**
   Every module since P5.1 has had to say "an antagonist alone is inert
   because P6.2 has not supplied endogenous transmitter"; [Britt2008]
   measured that inertness in tissue that certainly had endogenous dynorphin
   in it. The model agrees with the experiment for the wrong reason, and
   that distinction is recorded rather than glossed.

The weakness, stated first because it is the largest: BRL52537 is not
U50488. [Britt2008]'s 72% is the maximal suppression achievable THROUGH THE
RECEPTOR at saturating occupancy by a full agonist, and this module treats it
as a receptor-level ceiling shared by U50488 and dynorphin, both of which
[Toll1998] measured as full kappa agonists (93 +/- 11% and 97 +/- 5% of
U69,593). It is not a claim that U50488's own maximal effect was measured at
0.72. If a U50488 maximal suppression is published in a matched preparation,
`KOR_MAX_SUPPRESSED_FRACTION` is the one constant to change.

Sources
-------

Affinities and efficacies. ONE paper carries almost all of this module, and
finding it is most of what P5.7 was:

  [Toll1998]    Toll L, Berzetei-Gurske IP, Polgar WE, Brandt SR, Adapa ID,
                Rodriguez L, Schwartz RW, Haggart D, O'Brien A, White A,
                Kennedy JM, Craymer K, Farrington L, Auh JS. "Standard
                binding and functional assays related to medications
                development division testing for potential cocaine and
                opiate narcotic treatment medications." NIDA Res Monogr.
                1998;178:440-466. PMID:9686407. SRI International, under
                NIDA contracts N01DA-3-8302 and N01DA-4-8307.

                Table 5 reports Ki (nM, Cheng-Prusoff from competition
                curves with [3H]agonists, so Hill coefficients were near 1
                and the Ki designation is the authors' own) at HUMAN cloned
                mu-, delta- and kappa-opioid receptors expressed in CHO
                cells -- [3H]DAMGO at mu, [3H]DPDPE-Cl at delta,
                [3H]U69,593 at kappa -- and, in parallel columns, at the
                same three sites in guinea-pig brain membranes. Table 7
                reports [35S]GTPyS stimulation in membranes from the same
                three cell lines, as EC50 and as % of the site's own
                reference full agonist (DAMGO, DPDPE, U69,593
                respectively), which is exactly the normalization this
                codebase's `Ligand.efficacy` uses.

                It is a NIDA Research Monograph, i.e. a United States
                government publication in the public domain, and the full
                text including both tables is readable at
                archive.org/details/rms178. Every number below was read
                from that scan.

                Every one of the seven ligands used here was independently
                checked against the IUPHAR/BPS Guide to PHARMACOLOGY's own
                curation of this same paper (its human mu/delta/kappa
                interaction records citing PMID 9686407), and all
                twenty-one available pKi values agree to the rounding
                IUPHAR publishes. That is the provenance check on an OCR'd
                scan, and it is recorded because the alternative -- trusting
                character recognition on a 1998 typescript for numbers this
                module's every result depends on -- is not good enough.

Binding kinetics, one measured association rate per site:

  [Wang2021]    Wang X, Gou X, Yu X, Bai D, Tan B, Cao P, Qian M, Zheng X,
                Wang H, Tang P, Zhang C, Ye F, Ni J. "Antinociceptive and
                Antipruritic Effects of HSK21542, a Peripherally-Restricted
                Kappa Opioid Receptor Agonist, in Animal Models of Pain and
                Itch." Front Pharmacol. 2021;12:773204.
                doi:10.3389/fphar.2021.773204 (PMC8635029, open access).
                Human KOR-expressing cell membranes, 25 C,
                [3H]diprenorphine, Motulsky-Mahan competition association.
                The radioligand's own rate constants, measured from the
                association curves in that paper's Supplementary Figure S1
                and then held fixed in the competition fits, are
                k_on = 1.44 x 10^8 M^-1 min^-1 and k_off = 0.0257 min^-1.

  [Vaughn1989]  Vaughn LK, Knapp RJ, Toth G, Wan YP, Hruby VJ, Yamamura HI.
                "A high affinity, highly selective ligand for the delta
                opioid receptor: [3H]-[D-Pen2, pCl-Phe4, D-Pen5]enkephalin."
                Life Sci. 1989;45(11):1001-1008. PMID:2552241. RAT BRAIN
                homogenate, 25 C. Saturation Kd 328 +/- 27 pM; biphasic
                association, 5.05 +/- 2.5 x 10^8 and 0.147 +/- 0.014 x 10^8
                M^-1 min^-1; monophasic dissociation 2.96 +/- 0.25 x 10^-3
                min^-1; kinetic Kd 8.4 +/- 2.7 pM and 201 +/- 4 pM.

                This is the same radioligand [Toll1998] used for its delta
                column, which is why it was preferred over any other delta
                kinetics paper, and the two agree independently: Vaughn's
                saturation Kd of 0.328 nM against Toll's Ki of 0.3 nM for
                unlabelled DPDPE-Cl at delta-CHO, from a different
                laboratory, a different species and a different preparation.

Mechanism and anatomy (no binding number is taken from any of these):

  [DiChiara1988] Di Chiara G, Imperato A. "Opposite effects of mu and kappa
                opiate agonists on dopamine release in the nucleus
                accumbens and in the dorsal caudate of freely moving rats."
                J Pharmacol Exp Ther. 1988;244(3):1067-1080. PMID:2855239.
                Freely moving RATS, brain dialysis with HPLC-ED. Mu agonists
                (morphine, methadone, fentanyl) stimulated accumbal dopamine
                release; U-50,488, bremazocine and tifluadom DECREASED it in
                accumbens and caudate and reduced motor activity. The
                opposite-valence result this module's scenario has to
                reproduce, measured with two of this module's own drugs.
  [Devine1993]  Devine DP, Leone P, Pocock D, Wise RA. "Differential
                involvement of ventral tegmental mu, delta and kappa opioid
                receptors in modulation of basal mesolimbic dopamine
                release: in vivo microdialysis studies." J Pharmacol Exp
                Ther. 1993;266(3):1236-1246. PMID:7690399. Chloral-hydrate-
                anaesthetized RATS. Intra-VTA DAMGO and DPDPE each raised
                ventral striatal dopamine dose-orderly, DPDPE at 100- to
                1000-fold higher concentrations; naltrindole blocked DPDPE's
                effect and not DAMGO's, CTOP blocked DAMGO's and not
                DPDPE's; intra-VTA U-50,488H did nothing at any dose while
                systemic U-50,488H lowered dopamine. The reason delta goes
                on VTA_GABA and kappa does not.
  [Svingos2001] Svingos AL, Chavkin C, Colago EE, Pickel VM. "Major
                coexpression of kappa-opioid receptors and the dopamine
                transporter in nucleus accumbens axonal profiles." Synapse.
                2001;42(3):185-192. doi:10.1002/syn.10005. PMID:11746715.
                RAT accumbens, immuno-electron microscopy. Kappa labelling
                mainly presynaptic; of 400 kappa-labelled presynaptic
                structures 51% expressed the dopamine transporter. The
                ultrastructural reason kappa is a connection modulator here.
  [Britt2008]   Britt JP, McGehee DS. "Presynaptic opioid and nicotinic
                receptor modulation of dopamine overflow in the nucleus
                accumbens." J Neurosci. 2008;28(7):1672-1681.
                doi:10.1523/JNEUROSCI.4275-07.2008 (PMC6671549, open
                access). RAT striatal slices, P60-P90, fast electrochemistry.
                Saturating BRL52537 (1 uM) cut single-pulse-evoked dopamine
                overflow to 28 +/- 5% of control throughout dorsal and
                ventral striatum, not overcome by burst stimuli; nor-BNI
                (1 uM) mostly restored it and alone had no effect. The
                source of `KOR_MAX_SUPPRESSED_FRACTION`.
  [Margolis2003] Margolis EB, Hjelmstad GO, Bonci A, Fields HL.
                "Kappa-opioid agonists directly inhibit midbrain
                dopaminergic neurons." J Neurosci. 2003;23(31):9981-9986.
                doi:10.1523/JNEUROSCI.23-31-09981.2003 (PMC6740851). RAT VTA
                slices, whole-cell patch clamp. U69593 (1 uM) directly
                inhibits TH-immunoreactive VTA neurons via GIRK, blocked by
                nor-BNI (100 nM); some principal neurons showed both mu
                disinhibition and kappa inhibition. Cited for the mechanism
                this module deliberately does NOT build, so that the choice
                is visible.
  [Volpe2011]   Already cited in `scenarios.py` for the calibrated mu site.
                Reused here for one purpose only: the cross-assay check that
                licenses putting a [Toll1998] kappa Ki and a [Volpe2011] mu
                Ki in the same simulation. See `mu_assay_agreement()`.

Derivations
-----------

1. **Both constants measured** -- not available for any ligand here.
2. **koff and an affinity measured, kon derived** -- how the existing mu
   ligands were built (`scenarios.mu_opioid_ligands_calibrated()`), not used
   in this module.
3. **Only an affinity measured** -- share the site's measured `kon` and set
   `koff = kon * Kd`. Every ligand at both new sites.

Derivation 3 here is site-wise rather than chemotype-wise, and the reason is
that the measured rate available at each site belongs to that site's own
radioligand rather than to a congener of any modelled ligand:

  * At KAPPA every ligand takes [3H]diprenorphine's measured
    `kon = 0.144 nM^-1 min^-1` ([Wang2021]). The internal check is that
    diprenorphine's kinetic Kd from that same pair of constants,
    0.0257 / 0.144 = 0.178 nM, lands within 1.1-fold of [Toll1998]'s
    equilibrium Ki for diprenorphine at kappa-CHO, 0.2 nM -- two
    laboratories, two assay formats, one number.
  * At DELTA every ligand takes [3H]pCl-DPDPE's measured SLOW-component
    `kon = 0.0147 nM^-1 min^-1` ([Vaughn1989]). The fast component
    (0.505 nM^-1 min^-1) is recorded and not used: its kinetic Kd is 5.9 pM
    when recomputed as `koff/kon` from the paper's own printed constants
    (the paper reports 8.4 +/- 2.7 pM for it, so the recomputation lands
    inside its stated uncertainty), 56-fold tighter than the same paper's
    saturation Kd of 328 pM, while the slow component's 201 pM -- which the
    same recomputation reproduces exactly -- is within 1.6-fold of it.
    Choosing the
    component whose kinetic Kd reproduces the equilibrium Kd is the only
    non-arbitrary rule available, and it is stated rather than silently
    applied.

What derivation 3 costs, in the terms P5.3 and P5.6 established: the
AFFINITIES are exact by construction, because `koff = kon * Kd` reproduces
whatever Kd is passed in whatever the kon; the TIMESCALES on which occupancy
approaches those affinities are the radioligand's rather than the ligand's,
and a peptide and a small molecule at the same site plainly do not associate
at the same rate. Nothing this module concludes is stated in terms of a rate.
The 10-fold gap between the kappa and delta `kon` values is a gap between two
radioligands in two preparations and should not be read as a property of the
two receptors.

The censored affinities, which are bounds and are used as bounds
----------------------------------------------------------------

[Toll1998] reports ">10,000" rather than a number in two cells this module
needs: U50488 at delta, and Leu-enkephalin at kappa. Ten thousand nanomolar
is the top of the assay, not a measurement, and the codebase's first rule
forbids inventing what is behind it. The rule applied instead:

    a censored ">10,000" is instantiated AT the bound, which makes the
    modelled cross-activity an UPPER bound on the real cross-activity.

Setting Kd to the smallest value consistent with the datum maximises the
cross-talk the model can show. So when a test asserts that U50488 does
essentially nothing at delta, that conclusion is conservative: the real
number is somewhere above 10,000 nM and the real effect is somewhere below
the modelled one. `ASSAY_CEILING_NM` is the constant, `is_censored()` says
which cells it applies to, and a test asserts that no conclusion in this
module depends on the difference between "10,000" and "more than that".

Three things the numbers say that the folk summary does not
-----------------------------------------------------------

1. **Naloxone is 1.8-fold mu-over-kappa selective** (1.4 against 2.5 nM),
   and 48-fold mu-over-delta. Reversal of a mu agonist by naloxone is
   simultaneous, near-complete blockade of kappa. In THIS model that has no
   consequence at all when naloxone is given alone, because P6.2 has not
   supplied the endogenous dynorphin there would be to block -- and the
   moment dynorphin IS dosed, the model shows the consequence directly:
   naloxone reverses the kappa effect at a concentration within a factor of
   two of the one that reverses the mu effect. A test measures that ratio.
2. **Dynorphin A is only 4.5-fold kappa-selective over mu** (1.7 against
   7.7 nM) and is a FULL agonist at all three receptors in the same GTPyS
   panel (97%, 99%, 99%). The endogenous "kappa peptide" is barely a
   selective ligand at equilibrium. Its selectivity in vivo is a matter of
   where it is released, which is a Phase 6 property this model does not
   have.
3. **Leu-enkephalin binds mu tightly (7.4 nM) and barely activates it**
   (11 +/- 4% of DAMGO), while at delta it binds at 2.1 nM and is a full
   agonist (104 +/- 21% of DPDPE). Affinity and efficacy point in different
   directions for the same peptide at the same receptor, which is precisely
   what a model carrying only Ki would get wrong, and it is the reason this
   module takes both columns from [Toll1998] rather than assuming intrinsic
   activity from the ligand's usual label.

And one that the numbers say about the antagonists this module is asked for:
**nor-BNI is 28-fold kappa-over-delta selective and naltrindole is 32-fold
delta-over-mu**, both far short of the "highly selective" they are routinely
described as at equilibrium. nor-BNI's famous selectivity in vivo is a
duration phenomenon -- days to weeks of kappa antagonism after one dose --
which is a property of dissociation and receptor turnover that this module's
`TOOL_COMPOUND_KE` and derivation-3 `koff` do not represent, and which is
listed below under what was deliberately not built.

Two preparation disagreements, recorded rather than averaged
------------------------------------------------------------

[Toll1998] gives both a human-CHO and a guinea-pig-brain column for the same
twenty-one ligand/receptor cells, which is an unusually direct measurement of
how much a preparation is worth. Nineteen of the twenty-one agree within
3.5-fold. Two do not:

  * **naltrindole at MU**, 6.3 +/- 2.3 nM in mu-CHO against 0.2 +/- 0.01 nM
    in guinea-pig brain, 31.5-fold. That single cell is the difference
    between "naltrindole is 32-fold delta-selective" and "naltrindole is
    2.2-fold delta-selective".
  * **dynorphin A at DELTA**, 42.7 +/- 8.6 against 5.8 +/- 0.8 nM, 7.4-fold.

Worth recording alongside them: the KAPPA column is the one the two
preparations agree on best, every cell within 1.8-fold, which is the column
this module's headline results actually rest on.

This module uses the human-CHO column throughout, for three reasons stated so
the choice can be argued with: all seven ligands have a value in it; the
three receptors in it are matched recombinant cell lines rather than a tissue
in which the three sites differ in density and in G-protein environment; and
IUPHAR's independent curation of this paper records the 6.3 nM figure, so it
is the value the field carries. Both columns are in `KI_GUINEA_PIG_BRAIN` and
a test asserts the disagreements rather than hiding them.

Searched and empty, or found and deliberately not used
------------------------------------------------------

* **Binding kinetics for any of the seven ligands at kappa or delta.**
  Searched Europe PMC and PubMed for each ligand crossed with association /
  dissociation rate constant, residence time, competition association and
  Motulsky-Mahan. What exists is radioligand kinetics ([Wang2021],
  [Vaughn1989], both used) and biphasic-kinetics characterisations that
  report no usable rate constants for unlabelled ligands (Frances et al.'s
  [3H]U69593 analysis in guinea-pig cortex, PMID:2723659, is explicitly
  biphasic and gives numbers for neither component in its abstract; the
  [3H]naltrindole characterisation, PMID:8113996, reports saturation Kd only
  and states no rate constants). Hence derivation 3 at both sites.
* **A rodent kappa dose-response usable as a circuit-layer target.**
  [DiChiara1988] and [Devine1993] are both dose-based (mg/kg or intra-VTA
  nmol) and `drug_pk.py` has no U50488 kinetics, which is the same blocker
  Track E and P2.2 established for every other circuit gain in this project.
  [Britt2008] escapes it only because a bath concentration in a slice needs
  no pharmacokinetic bridge at all, which is why the kappa gain could be
  measured and the delta gain could not.
* **Rodent pharmacokinetics for U50488, nor-BNI, naltrindole, dynorphin or
  Leu-enkephalin.** Not adopted, for P5.3's two reasons: a real half-life
  paired with an illustrative gain buys nothing, and it would make an
  illustrative number look calibrated. `TOOL_COMPOUND_KE` is shared with
  every other P5 module and a test asserts the equality.
* **A measured maximal suppression for U50488 itself** at the accumbal
  dopamine terminal, in a preparation matched to [Britt2008]'s. Not found;
  BRL52537's is used as a receptor-level ceiling and the substitution is
  flagged above and asserted by a test on the docstring.

What was deliberately not built
-------------------------------

* **The somatodendritic kappa receptor on VTA dopamine neurons**
  ([Margolis2003]). Real, and a second, oppositely-placed kappa mechanism.
  Modelling both at once would need a way to say how much of the in vivo
  dopamine decrease each contributes, and [Devine1993]'s intra-VTA null
  result says the terminal one carries it. One mechanism, chosen by an
  experiment, with the other named.
* **The mu and delta receptors on accumbal cholinergic interneurons.**
  [Britt2008] found in the same slices that mu and delta agonists ALSO
  inhibit single-pulse-evoked dopamine overflow, via cholinergic interneuron
  inhibition, and that this inhibition IS overcome by burst stimulation.
  That is a mu action at the terminal opposite in sign to the mu action in
  the VTA, and it is frequency-dependent, so representing it needs both a
  cholinergic interneuron node this network does not have and a distinction
  between tonic and burst firing this mean-field abstraction cannot make.
  Its absence means this module's fentanyl effect on accumbal dopamine is
  larger than the two mechanisms together would give.
* **nor-BNI's duration.** One dose antagonises kappa for days to weeks in
  vivo. This module gives it `TOOL_COMPOUND_KE` and a derivation-3 `koff`,
  so its action here lasts tens of minutes. Any statement about how long a
  kappa blockade lasts is outside what is built.
* **Delta receptors outside the VTA.** Delta is dense in cortex and
  striatum, and `data/expression/aba_ish_unionize_Oprd1.csv` and
  `Oprk1.csv` are already in the repository, so the P4.2 factorization
  `coupling = mechanism_gain * rho` could be applied to both new receptors.
  It is not applied here: P5.7's brief is the constants and the
  cross-affinities, the delta gain has no anatomical spread to distribute
  because [Devine1993] localises the effect to one node, and the kappa
  mechanism is a connection modulator for which `rho` has no defined
  meaning. Named as the obvious next step rather than done badly.
* **Endogenous dynorphin and enkephalin tone.** The standing P6.2 gap, and
  here it is the whole content of the naloxone result: naloxone's kappa
  affinity matters clinically because there is dynorphin to displace, and
  this model has none until dynorphin is dosed by hand. The tests dose it by
  hand and say why.
* **Kappa-receptor-mediated dysphoria as a named behavioural endpoint.**
  The readouts stay `incentive_salience` etc., inherited from Track A, and
  the module claims a sign on a circuit-level quantity, not a behaviour.
  Phase 7 names behaviours.
"""

import numpy as np

from .behavior import BehaviorReadout
from .engine import ReceptorSystem
from .pk import OneCompartmentPK
from .receptor import Ligand, ReceptorSite
from .scenarios import multi_region_mu_opioid_network


#: The three receptors, in the order [Toll1998]'s tables use.
RECEPTORS = ("mu", "delta", "kappa")

#: The seven ligands this module builds, in the order they sit in every
#: receptor site below (and therefore in `pk_models_*()` and in the flat
#: `doses` vector). Agonists first, then the antagonists, then the two mu
#: ligands whose cross-affinities are the point of the task.
LIGANDS = (
    "dynorphin_A",
    "leu_enkephalin",
    "U50488",
    "nor_BNI",
    "naltrindole",
    "fentanyl",
    "naloxone",
)

#: nM. The top of [Toll1998]'s competition-binding assay. Two cells in the
#: table read ">10,000" rather than a number, and this is that number. It is
#: a BOUND: instantiating a censored affinity here maximises the modelled
#: cross-activity, so every "this ligand does nothing at that receptor"
#: conclusion in this module is conservative. See the module docstring.
ASSAY_CEILING_NM = 10000.0


# ---------------------------------------------------------------------------
# Affinities: [Toll1998] Table 5
# ---------------------------------------------------------------------------

#: nM. Ki at HUMAN cloned mu / delta / kappa receptors expressed in CHO
#: cells, [Toll1998] Table 5, columns "mu in mu-CHO [3H]DAMGO", "delta in
#: delta-CHO [3H]DPDPE-Cl" and "kappa in kappa-CHO [3H]U69,593". Cheng-Prusoff
#: Ki from competition curves against [3H]agonists. Values as printed,
#: including the two censored cells, which are stored at `ASSAY_CEILING_NM`
#: and flagged by `is_censored()`.
#:
#: Reported +/- values from the same table, kept in `KI_HUMAN_CHO_SD`.
KI_HUMAN_CHO = {
    "dynorphin_A": {"mu": 7.7, "delta": 42.7, "kappa": 1.7},
    "leu_enkephalin": {"mu": 7.4, "delta": 2.1, "kappa": ASSAY_CEILING_NM},
    "U50488": {"mu": 290.0, "delta": ASSAY_CEILING_NM, "kappa": 0.2},
    "nor_BNI": {"mu": 21.0, "delta": 5.7, "kappa": 0.2},
    "naltrindole": {"mu": 6.3, "delta": 0.2, "kappa": 10.1},
    "fentanyl": {"mu": 0.7, "delta": 152.7, "kappa": 84.8},
    "naloxone": {"mu": 1.4, "delta": 67.5, "kappa": 2.5},
}

#: nM. The standard deviations [Toll1998] prints beside each Ki above.
#: `None` where the cell is censored and has no dispersion to report.
KI_HUMAN_CHO_SD = {
    "dynorphin_A": {"mu": 2.2, "delta": 8.6, "kappa": 0.85},
    "leu_enkephalin": {"mu": 0.45, "delta": 0.4, "kappa": None},
    "U50488": {"mu": 14.3, "delta": None, "kappa": 0.05},
    "nor_BNI": {"mu": 5.0, "delta": 0.9, "kappa": 0.05},
    "naltrindole": {"mu": 2.3, "delta": 0.05, "kappa": 0.65},
    "fentanyl": {"mu": 0.3, "delta": 38.3, "kappa": 19.4},
    "naloxone": {"mu": 0.05, "delta": 40.0, "kappa": 0.3},
}

#: nM. The SAME table's guinea-pig brain membrane columns, same radioligands.
#: Not used to build any ligand. Recorded because six of the seven agree with
#: the human-CHO column within about 3-fold and the seventh, naltrindole at
#: mu, does not: 0.2 nM here against 6.3 nM there. See the module docstring
#: for why the human-CHO column is the one used, and
#: `preparation_disagreement()` for the quantity a test asserts.
KI_GUINEA_PIG_BRAIN = {
    "dynorphin_A": {"mu": 8.1, "delta": 5.8, "kappa": 1.7},
    "leu_enkephalin": {"mu": 21.7, "delta": 1.6, "kappa": ASSAY_CEILING_NM},
    "U50488": {"mu": 294.0, "delta": ASSAY_CEILING_NM, "kappa": 0.2},
    "nor_BNI": {"mu": 8.3, "delta": 6.3, "kappa": 0.3},
    "naltrindole": {"mu": 0.2, "delta": 0.09, "kappa": 7.8},
    "fentanyl": {"mu": 1.0, "delta": 73.8, "kappa": 151.2},
    "naloxone": {"mu": 1.5, "delta": 19.8, "kappa": 3.8},
}

#: nM. Diprenorphine's Ki in the same three human-CHO columns, [Toll1998]
#: Table 5. Not used to build a ligand: it is the check on derivation 3 at
#: kappa, where diprenorphine is the radioligand whose measured `kon` every
#: kappa ligand shares. Its kinetic Kd from [Wang2021] is 0.178 nM.
KI_DIPRENORPHINE_HUMAN_CHO = {"mu": 0.8, "delta": 0.5, "kappa": 0.2}

#: nM. DPDPE-Cl's Ki at delta-CHO in the same table, the same kind of check
#: at the delta site: [Vaughn1989] measured this radioligand's saturation Kd
#: at 0.328 nM in rat brain, a different laboratory and species.
KI_PCL_DPDPE_DELTA_CHO = 0.3

#: nM. [Volpe2011]'s mu Ki for the two ligands the calibrated mu site is
#: built from, repeated here for one purpose: the cross-assay agreement check
#: in `mu_assay_agreement()`. See `scenarios.py` for the full citation.
VOLPE_KI_MU = {"fentanyl": 1.346, "naloxone": 1.518}


def is_censored(ligand, receptor, preparation="human_cho"):
    """True where [Toll1998] printed ">10,000" instead of a number.

    Two cells: U50488 at delta and Leu-enkephalin at kappa, in both
    preparations. The stored value is the bound itself; see the module
    docstring for why instantiating at the bound makes every conclusion
    drawn from those cells conservative.
    """
    return ki(ligand, receptor, preparation) >= ASSAY_CEILING_NM


def _table(preparation):
    if preparation == "human_cho":
        return KI_HUMAN_CHO
    if preparation == "guinea_pig_brain":
        return KI_GUINEA_PIG_BRAIN
    raise ValueError(
        "preparation must be 'human_cho' or 'guinea_pig_brain', got %r"
        % (preparation,)
    )


def ki(ligand, receptor, preparation="human_cho"):
    """Ki in nM, [Toll1998] Table 5. Default the human cloned-receptor column."""
    table = _table(preparation)
    if ligand not in table:
        raise ValueError("unknown ligand %r; have %s" % (ligand, sorted(table)))
    if receptor not in RECEPTORS:
        raise ValueError("receptor must be one of %s, got %r" % (list(RECEPTORS), receptor))
    return table[ligand][receptor]


def selectivity(ligand, preferred, over, preparation="human_cho"):
    """`Ki(over) / Ki(preferred)`: how many fold this ligand prefers one
    receptor to another. 1.0 means no selectivity at all.

    A ratio of two numbers from the same table, the same assay and the same
    day is the only selectivity statement [Toll1998] supports, and it is the
    reason a single-laboratory panel was worth the search: a Ki for mu from
    one paper divided by a Ki for kappa from another is a comparison of two
    laboratories as much as of two receptors.
    """
    return ki(ligand, over, preparation) / ki(ligand, preferred, preparation)


def naloxone_mu_over_kappa_selectivity(preparation="human_cho"):
    """1.79 in human CHO, 2.53 in guinea-pig brain.

    The number the brief exists for. Naloxone is not a mu-selective
    antagonist: on the human cloned receptors it prefers mu to kappa by less
    than two-fold, which is within the spread of a single competition curve.
    Any naloxone exposure that blocks mu has blocked kappa too.
    """
    return selectivity("naloxone", "mu", "kappa", preparation)


def fentanyl_mu_selectivity(preparation="human_cho"):
    """`{'delta': 218.1, 'kappa': 121.1}` in human CHO.

    Real selectivity, and finite. Two hundred-fold is not infinite: at a
    concentration two hundred times the one that saturates mu, fentanyl is
    half-occupying delta, and the efficacy table says it is a near-full
    agonist there and a 30% partial agonist at kappa.
    """
    return {
        receptor: selectivity("fentanyl", "mu", receptor, preparation)
        for receptor in ("delta", "kappa")
    }


def preparation_disagreement(ligand, receptor):
    """`Ki(guinea-pig brain) / Ki(human CHO)`, or its reciprocal if that is
    larger: the fold disagreement between [Toll1998]'s own two columns.

    Six of the seven ligands sit under about 3 at every receptor. Naltrindole
    at mu sits at 31.5, and that single cell is the difference between
    "naltrindole is 32-fold delta-selective" and "naltrindole is 2.2-fold
    delta-selective". A test asserts both halves.
    """
    a = ki(ligand, receptor, "human_cho")
    b = ki(ligand, receptor, "guinea_pig_brain")
    ratio = b / a
    return ratio if ratio >= 1.0 else 1.0 / ratio


def mu_assay_agreement():
    """`Ki_Toll / Ki_Volpe` at mu for fentanyl and naloxone.

    The check that licenses this module at all. The existing calibrated mu
    site is built from [Volpe2011]'s standardized [3H]DAMGO assay on
    recombinant human MOR (fentanyl 1.346 nM, naloxone 1.518 nM); the kappa
    and delta sites below are built from [Toll1998]'s human CHO panel, whose
    own mu column reads 0.7 and 1.4 nM. The returned ratios are 0.52 and
    0.92, i.e. the two laboratories differ by 1.9-fold for fentanyl and by
    1.08-fold for naloxone. Two independent groups two decades apart
    agreeing that closely on the same two ligands at the same receptor is
    what makes it defensible to put a [Toll1998] kappa Ki and a [Volpe2011]
    mu Ki into one simulation.

    It is not a licence to mix arbitrarily: the delta and kappa numbers have
    no second laboratory behind them here, and `receptor_site_mu()` exists so
    that a fully single-assay configuration can be run and compared. P5.7's
    report does compare them.
    """
    return {
        name: KI_HUMAN_CHO[name]["mu"] / value for name, value in VOLPE_KI_MU.items()
    }


# ---------------------------------------------------------------------------
# Efficacies: [Toll1998] Table 7
# ---------------------------------------------------------------------------

#: Fraction of the site's own reference full agonist's maximal [35S]GTPyS
#: stimulation, [Toll1998] Table 7, human mu-/delta-/kappa-CHO membranes.
#: The reference is DAMGO at mu, DPDPE at delta and U69,593 at kappa, and
#: "% Stimulation compared to DAMGO, DPDPE or U69,593, respectively" is the
#: table's own footnote -- which is exactly the normalization
#: `Ligand.efficacy` means, so these are used unconverted.
#:
#: `0.0` is written where the table prints "flat", i.e. no measurable
#: stimulation at any concentration tested. THAT IS A MEASUREMENT, and it is
#: the first time in this codebase that a neutral-antagonist efficacy of 0.0
#: is one: every module from P5.1 to P5.6 had to flag `efficacy=0.0` as a
#: modelling simplification standing in for an unmeasured inverse agonism.
#: Here naloxone, nor-BNI and naltrindole were each run in the functional
#: assay at all three receptors and each was flat at all three.
#:
#: Values above 1.0 are as printed and are not clipped: Leu-enkephalin at
#: delta is 104 +/- 21% of DPDPE, i.e. indistinguishable from the reference
#: and printed as slightly above it. Rounding it down to 1.0 would be
#: replacing a measurement with a tidier number.
EMAX_GTPGS = {
    "dynorphin_A": {"mu": 0.99, "delta": 0.99, "kappa": 0.97},
    "leu_enkephalin": {"mu": 0.11, "delta": 1.04, "kappa": 0.89},
    "U50488": {"mu": 0.0, "delta": 0.0, "kappa": 0.93},
    "nor_BNI": {"mu": 0.0, "delta": 0.0, "kappa": 0.0},
    "naltrindole": {"mu": 0.0, "delta": 0.0, "kappa": 0.0},
    "fentanyl": {"mu": 1.00, "delta": 0.86, "kappa": 0.30},
    "naloxone": {"mu": 0.0, "delta": 0.0, "kappa": 0.0},
}

#: The ligand/receptor pairs [Toll1998] Table 7 printed as "flat": measured
#: and not stimulating, as opposed to not measured. Kept separately from the
#: zeros above so that "this efficacy is a measurement" is checkable.
MEASURED_FLAT = frozenset(
    [
        ("U50488", "mu"),
        ("U50488", "delta"),
        ("nor_BNI", "mu"),
        ("nor_BNI", "delta"),
        ("nor_BNI", "kappa"),
        ("naltrindole", "mu"),
        ("naltrindole", "delta"),
        ("naltrindole", "kappa"),
        ("naloxone", "mu"),
        ("naloxone", "delta"),
        ("naloxone", "kappa"),
    ]
)


def efficacy(ligand, receptor):
    """Intrinsic activity at one receptor, [Toll1998] Table 7."""
    if ligand not in EMAX_GTPGS:
        raise ValueError("unknown ligand %r; have %s" % (ligand, sorted(EMAX_GTPGS)))
    if receptor not in RECEPTORS:
        raise ValueError("receptor must be one of %s, got %r" % (list(RECEPTORS), receptor))
    return EMAX_GTPGS[ligand][receptor]


def fentanyl_is_a_partial_kappa_agonist():
    """`(0.30, 1.00)`: fentanyl's kappa and mu efficacies.

    The half of "fentanyl is not mu-selective in the limit" that a Ki table
    cannot show. Fentanyl at kappa is not an inert occupier: it is a 30%
    partial agonist of a receptor whose action opposes its own mu action, so
    at high enough exposure it is working against itself. In this model that
    is an occupancy question with a measured answer, and a test measures how
    much of fentanyl's own incentive-salience effect its kappa agonism eats.
    """
    return efficacy("fentanyl", "kappa"), efficacy("fentanyl", "mu")


# ---------------------------------------------------------------------------
# Kinetics
# ---------------------------------------------------------------------------

#: nM^-1 min^-1. MEASURED association rate constant of [3H]diprenorphine at
#: the HUMAN kappa receptor, [Wang2021]: 1.44e8 M^-1 min^-1, i.e. 1.44e8 *
#: 1e-9. Shared by every kappa ligand (derivation 3).
KAPPA_KON_DIPRENORPHINE = 0.144

#: min^-1. The MEASURED dissociation rate constant of the same radioligand in
#: the same fit, 0.0257 min^-1. Not used to build a ligand: it is the check
#: on derivation 3 at this site (see `kappa_kinetic_check()`).
KAPPA_KOFF_DIPRENORPHINE = 0.0257

#: nM^-1 min^-1. MEASURED association rate constant of [3H]pCl-DPDPE at the
#: rat brain delta receptor, [Vaughn1989], SLOW component: 0.147e8 M^-1
#: min^-1. Shared by every delta ligand (derivation 3). See the module
#: docstring for why the slow component and not the fast one.
DELTA_KON_PCL_DPDPE = 0.0147

#: nM^-1 min^-1. The FAST component from the same biphasic fit, 5.05e8 M^-1
#: min^-1. Recorded and deliberately not used. Its kinetic Kd, recomputed as
#: `koff/kon` from the same paper's printed constants, is 5.9 pM (the paper
#: reports 8.4 +/- 2.7 pM, so the recomputation is inside its stated
#: uncertainty), 56 times tighter than that paper's saturation Kd of 328 pM.
DELTA_KON_PCL_DPDPE_FAST = 0.505

#: min^-1. The MEASURED, monophasic dissociation rate constant of
#: [3H]pCl-DPDPE, 2.96e-3 min^-1. The check on derivation 3 at delta.
DELTA_KOFF_PCL_DPDPE = 0.00296

#: nM. [Vaughn1989]'s own saturation Kd for that radioligand, 328 pM, and the
#: quantity the slow component's kinetic Kd (201 pM) has to reproduce for the
#: component choice above to be the right one.
DELTA_KD_PCL_DPDPE_SATURATION = 0.328


def kappa_kinetic_check():
    """`(kinetic Kd, equilibrium Ki)` for diprenorphine at kappa.

    `0.0257 / 0.144 = 0.178 nM` from [Wang2021]'s kinetics against
    [Toll1998]'s equilibrium Ki of 0.2 nM, two laboratories and two assay
    formats apart. 1.1-fold. This is what makes sharing that `kon` across
    the kappa site a derivation rather than a guess.
    """
    kinetic = KAPPA_KOFF_DIPRENORPHINE / KAPPA_KON_DIPRENORPHINE
    return kinetic, KI_DIPRENORPHINE_HUMAN_CHO["kappa"]


def delta_kinetic_check():
    """`(kinetic Kd slow, kinetic Kd fast, saturation Kd)` at delta, in nM.

    0.201, 0.0059 and 0.328. The slow component reproduces the equilibrium
    number to 1.6-fold and the fast one misses it by 56-fold, which is the
    stated rule for choosing between them.
    """
    slow = DELTA_KOFF_PCL_DPDPE / DELTA_KON_PCL_DPDPE
    fast = DELTA_KOFF_PCL_DPDPE / DELTA_KON_PCL_DPDPE_FAST
    return slow, fast, DELTA_KD_PCL_DPDPE_SATURATION


SITE_KON = {"kappa": KAPPA_KON_DIPRENORPHINE, "delta": DELTA_KON_PCL_DPDPE}


# ---------------------------------------------------------------------------
# Pharmacokinetics
# ---------------------------------------------------------------------------

#: min^-1. ILLUSTRATIVE. ln(2) / 30 min, deliberately the same number as
#: `serotonin.TOOL_COMPOUND_KE`, `dopamine.TOOL_COMPOUND_KE`,
#: `noradrenaline.TOOL_COMPOUND_KE`, `acetylcholine.TOOL_COMPOUND_KE`,
#: `histamine.TOOL_COMPOUND_KE` and `adenosine.TOOL_COMPOUND_KE`; a test
#: asserts the equality. Carried by U50488, nor-BNI and naltrindole. Do not
#: cite it as any of their half-lives in any species, and note especially
#: that nor-BNI's in vivo kappa antagonism lasts days to weeks, which this
#: cannot represent.
TOOL_COMPOUND_KE = 0.023105

#: min^-1. ILLUSTRATIVE PROXY for dynorphin A and Leu-enkephalin, the same
#: value and the same reason as `dopamine`'s and `adenosine`'s transmitter
#: clearance constants. Extracellular opioid peptides are removed by
#: membrane peptidases (aminopeptidase N, neprilysin) on a
#: seconds-to-minutes timescale, which is enzymatic and saturable rather
#: than first-order elimination from a compartment. 5.0 min^-1 (t1/2 ~8.3 s)
#: preserves the only property the dynamics need -- that the peptide clears
#: far faster than a dosed small molecule -- at a timescale the integrator
#: resolves. P6.2 is where a real peptide tone belongs.
PEPTIDE_KE = 5.0

#: min^-1. Fentanyl and naloxone keep the CALIBRATED elimination rate
#: constants Track A derived for them, so that the same drug has the same
#: pharmacokinetics at all three of its receptors. See
#: `scenarios.mu_opioid_pk_models_calibrated()` for the derivation and its
#: caveats: fentanyl's is ln(2)/13 min, the redistribution-phase half-life;
#: naloxone's is ln(2)/64 min.
FENTANYL_KE = 0.05332
NALOXONE_KE = 0.01083

KE_BY_LIGAND = {
    "dynorphin_A": PEPTIDE_KE,
    "leu_enkephalin": PEPTIDE_KE,
    "U50488": TOOL_COMPOUND_KE,
    "nor_BNI": TOOL_COMPOUND_KE,
    "naltrindole": TOOL_COMPOUND_KE,
    "fentanyl": FENTANYL_KE,
    "naloxone": NALOXONE_KE,
}


def pk_models():
    """One PK model per ligand, in `LIGANDS` order.

    The same list is used for all three sites, because the same drug has the
    same kinetics wherever it binds. The engine gives each receptor system
    its own concentration state, so dosing `fentanyl` through
    `Simulation.dose_vector` sets every one of them identically and they
    stay identical: that is a duplicated integration, not a duplicated drug.
    """
    return [OneCompartmentPK(ke=KE_BY_LIGAND[name]) for name in LIGANDS]


# ---------------------------------------------------------------------------
# The sites
# ---------------------------------------------------------------------------


def _ligand(name, receptor, preparation="human_cho"):
    kon = SITE_KON.get(receptor, KAPPA_KON_DIPRENORPHINE)
    return Ligand(
        name=name,
        kon=kon,
        koff=kon * ki(name, receptor, preparation),
        efficacy=efficacy(name, receptor),
    )


def ligands_kappa(preparation="human_cho"):
    """All seven ligands at kappa, in `LIGANDS` order.

    Every one of them, including the two mu drugs and the delta antagonist,
    because "which ligands belong at this receptor" is a question the
    affinity table answers and the answer is all of them: the smallest
    cross-affinity in the column is Leu-enkephalin's, and that one is a
    censored bound rather than a measured absence.
    """
    return [_ligand(name, "kappa", preparation) for name in LIGANDS]


def ligands_delta(preparation="human_cho"):
    """All seven ligands at delta, in `LIGANDS` order."""
    return [_ligand(name, "delta", preparation) for name in LIGANDS]


def ligands_mu(preparation="human_cho"):
    """All seven ligands at mu, from [Toll1998] rather than from Track A.

    This is NOT the site the existing mu scenario uses, and it is not meant
    to replace it. `scenarios.mu_opioid_receptor_site_calibrated()` stays
    exactly as it is -- [Volpe2011] affinities, [Mann2022] measured `koff` --
    and remains the calibrated mu site of this codebase, because its two
    ligands each have a MEASURED dissociation rate constant and these do not.

    What this site is for is the single-assay control: a configuration in
    which all three receptors come from one laboratory's one panel, so that
    the cost of mixing [Volpe2011] with [Toll1998] can be measured instead of
    argued. `mu_assay_agreement()` says the two mu columns agree to 1.9-fold
    and 1.08-fold; P5.7's report runs both configurations and reports what
    that 1.9-fold does to the result.
    """
    return [_ligand(name, "mu", preparation) for name in LIGANDS]


def receptor_site_kappa(preparation="human_cho"):
    """One competitive kappa pool, on the accumbal dopaminergic terminal."""
    return ReceptorSite(
        name="kappa_opioid_on_mesolimbic_DA_terminal",
        ligands=ligands_kappa(preparation),
    )


def receptor_site_delta(preparation="human_cho"):
    """One competitive delta pool, on the VTA GABAergic interneuron."""
    return ReceptorSite(
        name="delta_opioid_on_VTA_GABA", ligands=ligands_delta(preparation)
    )


def receptor_site_mu(preparation="human_cho"):
    """One competitive mu pool built entirely from [Toll1998]; see
    `ligands_mu()` for why this exists alongside the calibrated site."""
    return ReceptorSite(
        name="mu_opioid_on_VTA_GABA", ligands=ligands_mu(preparation)
    )


# ---------------------------------------------------------------------------
# The circuit: reused, not built
# ---------------------------------------------------------------------------

#: The node the delta receptor sits on. [Devine1993]: intra-VTA DPDPE raises
#: ventral striatal dopamine, naltrindole blocks it, CTOP does not.
DELTA_NODE = "VTA_GABA"

#: The connection the kappa receptor modulates, in `(post, pre)` order as
#: `Simulation.connection_modulation` expects: the mesolimbic dopamine
#: projection from VTA dopamine neurons to the accumbens. [Svingos2001] puts
#: the receptor on that terminal; [Devine1993] rules out the cell body.
KAPPA_CONNECTION = ("NAc", "VTA_DA")


def reward_network():
    """`scenarios.multi_region_mu_opioid_network()`, unchanged.

    Deliberately the same object rather than a copy: the same seven nodes,
    weights, taus, gains and baseline inputs the mu-opioid scenario has used
    since Track C1. Kept as the regression baseline -- every mu result this
    codebase already has was produced on it -- and NOT as this module's
    default network, for the reason `stabilized_reward_network()` gives.
    """
    return multi_region_mu_opioid_network()


#: What `stabilize()` sets the post-scaling contraction number to. P3.4's
#: own default, repeated here so a reader does not have to open `stability`
#: to know that the certificate this module claims is the certified one.
OPERATING_GAIN = 0.9


def stabilized_reward_network():
    """The same network put through P3.4's own stabilization procedure.

    `stability.stabilize()` multiplies every weight by ONE global scalar
    (0.342 here, `rho / kappa` with `rho = 0.9` and the source network's
    contraction number 2.632) and then re-solves the seven `baseline_input`
    values so the drug-free fixed point is exactly 0.5 at every node.
    Relative weights, topology, taus and gains are untouched.

    Why this module cannot use Track C1's network unmodified
    -------------------------------------------------------

    This was not a stylistic preference, it was forced, and the way it was
    forced is worth recording because it is P3.4's saturation lesson
    reappearing in a new place.

    The kappa mechanism here is a PRESYNAPTIC modulator of the
    `VTA_DA -> NAc` projection: it scales what that projection transmits. On
    Track C1's hand-set baselines the drug-free fixed point of the reward
    block is `VTA_GABA = 0.987`, `VTA_DA = 4.9e-6`, `NAc = 0.500`,
    `PFC = 0.985` -- VTA_GABA is nearly saturated ON, which drives VTA_DA
    saturated OFF, which means the mesolimbic projection carries nothing at
    rest. **A presynaptic modulator of a synapse that is transmitting
    nothing has, exactly, no effect**: with U50488 at 100 nM and kappa
    occupancy above 99%, the endpoint `incentive_salience` moves by 0.0
    on that network. It is a property of the operating point, not of the
    pharmacology, and no choice of gain or dose fixes it.

    The intermediate option, `balanced_baselines` alone (re-balance the
    baselines, leave the weights), does not work either and fails in the
    other direction: the reward loop's contraction number is 2.632, and
    with the nodes moved into the sensitive part of their sigmoids the loop
    oscillates -- `stability.steady_state` returns residuals around 6e-2
    from most initial conditions, i.e. no fixed point at all. So of the
    three candidates only the fully stabilized one lets the question be
    asked: its contraction number is 0.9, and `stability.steady_state`
    returns all-0.5 with a residual around 1e-9 from every one of P3.4's
    default initial conditions, which is the global-uniqueness certificate
    the other two do not have.

    What it costs, stated plainly: this is not literally the network the
    existing mu results were computed on, so P5.7's re-run reports the
    cross-affinity comparison on BOTH -- on Track C1's network, where it is
    the strict regression against the published results, and on this one,
    where the kappa mechanism is expressible. See
    `results/opioid_cross_affinity.txt` for both tables.
    """
    from . import stability

    return stability.stabilize(
        multi_region_mu_opioid_network(), operating_gain=OPERATING_GAIN
    ).network


def rest_rates(network=None):
    """The drug-free fixed point of `stabilized_reward_network()`: 0.5
    everywhere, by construction of `balanced_baselines`.

    Passing `rates0=rest_rates()` makes a drug-free run exactly flat and
    every deviation in a dosed run the drug, the habit P5.2 onwards adopted.
    It matters more here than in most modules, because the effect this one
    exists to show is a change of a few hundredths in a readout and a
    start-up transient of the size Track C1's network has would bury it.
    """
    n = 7 if network is None else network.n
    return np.full(n, 0.5)


# ---------------------------------------------------------------------------
# Coupling
# ---------------------------------------------------------------------------

#: ILLUSTRATIVE. How much a unit of delta activation changes the drive of
#: the VTA GABAergic interneuron. NEGATIVE, and that sign is measured:
#: [Devine1993]'s intra-VTA DPDPE raised ventral striatal dopamine, which in
#: this network is the disinhibition VTA_GABA -| VTA_DA and therefore a
#: reduction of VTA_GABA's own drive.
#:
#: Magnitude 1.0 by P5.1's per-receptor unit rule, and NOT the -1.72 the mu
#: receptor carries on the same node. That is a deliberate refusal for
#: exactly A2A's reason: matching the magnitudes would look like the natural
#: encoding of "delta does what mu does" and would assert that a unit of
#: delta activation and a unit of mu activation move that interneuron
#: equally, which nothing measured supports. [Devine1993] reports that DPDPE
#: needed 100- to 1000-fold HIGHER concentrations than DAMGO, but that is a
#: statement about ligand potency, already carried by the Kd's below, not
#: about how much the receptors differ per unit of activation. No mu-versus-
#: delta potency comparison should be read out of this module and a test
#: says so.
#:
#: Not identifiable from anything published (P2.3), and it is the
#: `mechanism_gain` half of P4.2's `coupling = mechanism_gain * rho` with no
#: rho applied, because [Devine1993] localises the effect to one node.
DELTA_MECHANISM_GAIN = -1.0

#: The fraction of evoked accumbal dopamine overflow a SATURATING kappa
#: agonist removes: 1 - 0.28 = 0.72, from [Britt2008]'s "to 28 +/- 5% of
#: control" with 1 uM BRL52537 in rat striatal slices. MEASURED.
KOR_MAX_SUPPRESSED_FRACTION = 0.72

#: The `connection_modulation` gain. NEGATIVE and MEASURED: the engine forms
#: `1 + gain * activation`, so `gain = -KOR_MAX_SUPPRESSED_FRACTION` makes a
#: saturating full agonist leave the mesolimbic weight at 28% of its
#: strength and a drug-free run leave it at exactly 100%. Same construction
#: as `cannabinoid.CB1_CONNECTION_GAIN`, and the second measured gain in
#: this codebase.
KOR_CONNECTION_GAIN = -KOR_MAX_SUPPRESSED_FRACTION


def coupling_delta():
    """Delta activation -> drive, on the VTA GABAergic interneuron."""
    return {DELTA_NODE: DELTA_MECHANISM_GAIN}


def coupling_kappa():
    """Empty, and that is the mechanism claim.

    A presynaptic receptor on a dopaminergic terminal has no drive of its
    own: it scales the transmission of one identified synapse. Giving it a
    `coupling` entry on NAc would assert that a kappa agonist drives the
    accumbens directly, independently of whether any dopamine is being
    released, which is not what [Svingos2001] and [Britt2008] describe. Same
    argument, same shape, as `cannabinoid.coupling_cb1()`.
    """
    return {}


def connection_modulation_kappa():
    """`{(NAc, VTA_DA): -0.72}`. The measured maximal suppression."""
    return {KAPPA_CONNECTION: KOR_CONNECTION_GAIN}


def modulated_mesolimbic_weight(activation, base_weight=2.0):
    """The effective `W[NAc, VTA_DA]` at a given kappa activation.

    Written out so that "a kappa agonist cannot silence this projection, and
    cannot invert it either" is arithmetic a test can check rather than a
    sentence. `base_weight` defaults to the value
    `scenarios.multi_region_mu_opioid_network()` actually uses.
    """
    return base_weight * (1.0 + KOR_CONNECTION_GAIN * activation)


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------


def receptor_system_kappa(preparation="human_cho"):
    """The kappa stack as a `ReceptorSystem`: no coupling, one modulated
    connection."""
    return ReceptorSystem(
        pk_models=pk_models(),
        receptor_site=receptor_site_kappa(preparation),
        coupling=coupling_kappa(),
        connection_modulation=connection_modulation_kappa(),
        name="kappa_opioid",
    )


def receptor_system_delta(preparation="human_cho"):
    """The delta stack as a `ReceptorSystem`."""
    return ReceptorSystem(
        pk_models=pk_models(),
        receptor_site=receptor_site_delta(preparation),
        coupling=coupling_delta(),
        name="delta_opioid",
    )


def receptor_system_mu_toll(preparation="human_cho", coupling=None):
    """A mu system built from [Toll1998], for the single-assay control.

    Reuses the mu scenario's own coupling map unchanged, so the only thing
    that differs from the calibrated mu system is where the binding
    constants came from.
    """
    from . import scenarios

    return ReceptorSystem(
        pk_models=pk_models(),
        receptor_site=receptor_site_mu(preparation),
        coupling=scenarios.mu_opioid_multi_region_coupling() if coupling is None else coupling,
        name="mu_opioid_toll",
    )


def mu_receptor_system_calibrated(coupling=None):
    """The EXISTING calibrated mu system, on the multi-region coupling map.

    Nothing here is new: `scenarios.mu_opioid_receptor_site_calibrated()` and
    `scenarios.mu_opioid_pk_models_calibrated()` are Track A's, and the
    coupling is P4.3's atlas-derived map. It is assembled in this module only
    so that the mu-only and the mu-plus-kappa-plus-delta configurations are
    built by the same code path and differ in exactly one thing: the presence
    of the two new systems.
    """
    from . import scenarios

    return ReceptorSystem(
        pk_models=scenarios.mu_opioid_pk_models_calibrated(),
        receptor_site=scenarios.mu_opioid_receptor_site_calibrated(),
        coupling=scenarios.mu_opioid_multi_region_coupling() if coupling is None else coupling,
        name="mu_opioid",
    )


def mu_only_receptor_systems():
    """The existing mu scenario: one system, two ligands. The baseline the
    cross-affinity re-run is measured against."""
    return [mu_receptor_system_calibrated()]


def opioid_receptor_systems(preparation="human_cho"):
    """Mu (calibrated, unchanged) + kappa + delta on one shared network.

    The P5.7 configuration. Dosing `fentanyl` or `naloxone` through
    `Simulation.dose_vector` now reaches all three receptors at once, at
    their measured cross-affinities, which is the whole point.
    """
    return [
        mu_receptor_system_calibrated(),
        receptor_system_kappa(preparation),
        receptor_system_delta(preparation),
    ]


def single_assay_receptor_systems(preparation="human_cho"):
    """The same three receptors with every constant from [Toll1998].

    The control for `opioid_receptor_systems()`: it swaps the calibrated mu
    site for the Toll one, so the difference between the two runs is exactly
    the difference between two laboratories' mu measurements of the same two
    drugs.
    """
    return [
        receptor_system_mu_toll(preparation),
        receptor_system_kappa(preparation),
        receptor_system_delta(preparation),
    ]


def kappa_delta_receptor_systems(preparation="human_cho"):
    """The two new systems alone, without mu. Used to isolate what the
    kappa and delta receptors do to a readout with the mu contribution
    removed."""
    return [receptor_system_kappa(preparation), receptor_system_delta(preparation)]


# ---------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------


def incentive_salience_readout():
    """`scenarios.incentive_salience_readout()`, re-exported unchanged.

    Deliberately the same name, nodes and weights, and a test asserts the
    equality: the opposite-valence result this module exists to show is a
    statement about the readout the mu scenario has always used, not about a
    readout chosen to make it come out that way. It is a circuit-level
    quantity (0.7 * NAc + 0.3 * VTA_DA), and the sign it moves in under a
    kappa agonist is what [DiChiara1988] is cited for; its magnitude is not
    a behavioural claim.
    """
    return BehaviorReadout(
        name="incentive_salience", weights={"NAc": 0.7, "VTA_DA": 0.3}
    )


def accumbal_drive_readout():
    """NAc firing alone: the node the kappa mechanism actually acts on.

    Reported separately from `incentive_salience` because that readout mixes
    NAc with VTA_DA, and the kappa mechanism modulates the connection
    between them without touching VTA_DA's own drive. Separating the two
    makes it visible that the kappa effect arrives entirely through the
    projection and not through the cell body -- which is [Devine1993]'s
    result, restated as an identity of this model.
    """
    return BehaviorReadout(name="accumbal_drive", weights={"NAc": 1.0})


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def run(
    systems=None,
    network=None,
    t_end=60.0,
    n_points=241,
    method="LSODA",
    rates0=None,
    **doses
):
    """Integrate one configuration and return `(network, sim, result)`.

    A convenience shared by the tests, the example script and the report so
    that the three cannot drift apart in their integration settings.
    Defaults to `stabilized_reward_network()` started at its own fixed
    point, so a drug-free run is exactly flat; pass `network=` and
    `rates0=` explicitly to run on Track C1's original network instead.
    """
    from .engine import Simulation

    network = stabilized_reward_network() if network is None else network
    systems = opioid_receptor_systems() if systems is None else systems
    if rates0 is None:
        rates0 = rest_rates(network)
    sim = Simulation(network=network, receptor_systems=systems)
    result = sim.run(
        doses=sim.dose_vector(**doses),
        t_span=(0.0, t_end),
        t_eval=np.linspace(0.0, t_end, n_points),
        rates0=rates0,
        method=method,
    )
    return network, sim, result


def readout_trace(readout, network, result):
    """A readout evaluated at every time point of a run."""
    rates = result["rates"]
    return np.array(
        [readout.score(network, rates[:, i]) for i in range(rates.shape[1])]
    )


def final_score(readout, network, result):
    """The readout's value at the last time point of a run."""
    return readout.score(network, result["rates"][:, -1])
