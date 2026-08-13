"""P5.5: the glutamate NMDA and AMPA receptor systems.

Units, as everywhere else in this codebase: concentration in nM, time in
minutes, `kon` in nM^-1 min^-1, `koff` and `ke` in min^-1. Membrane
potentials are in mV and are NOT a state of this model; see "The membrane
potential this model does not have".

THE DOUBLE-COUNTING QUESTION, WHICH IS WHAT THIS MODULE IS ABOUT
================================================================

Every receptor system before this one adds DRIVE to a node: the receptor
activates, and its activation is added to that node's Wilson-Cowan input
term scaled by a coupling gain. Doing that for glutamate would be wrong,
and not subtly wrong.

`network.CircuitNetwork` integrates

    tau_i * dr_i/dt = -r_i + sigmoid(sum_j W[i,j] * r_j + drive_i)

`W[i,j]` is the net synaptic gain from population j onto population i. For a
POSITIVE (excitatory) entry, the synapses that gain is made of are
glutamatergic, and the receptors that transduce them are AMPA and NMDA
receptors. That is not an interpretation this module is imposing: it is what
`W[i,j] > 0` already means, in every network in this codebase --
`docs/architecture.md`'s "Network layer: connectivity as a signed matrix",
P3.2's derivation of excitatory/inhibitory sign from region
neurotransmitter identity, and every hand-built cortical loop from P5.1
onward, where the pyramidal-to-interneuron weight IS the glutamatergic
projection.

So fast glutamatergic transmission is ALREADY IN THE MODEL, at full
strength, in every simulation this codebase has ever run, and it was there
before this module existed. Adding a `coupling={"PFC_GABA": g}` term driven
by AMPA-receptor activation would add it a second time: the same synapse
would contribute `W[i,j] * r_j` and, independently, `g * activation`. The
model would then contain two glutamate synapses where the brain has one,
the resting firing rates would move for no physiological reason, and an AMPA
antagonist would only ever be able to remove ONE of them.

**The resolution.** Glutamate receptor systems in this module do not couple
to node drive at all. `coupling` is empty for every one of them, on purpose.
They enter the circuit exclusively through `connection_modulation`, which
scales an EXISTING connectivity weight::

    W_eff[i,j] = W[i,j] * (1 - share_NMDA * blocked_NMDA
                              - share_AMPA * blocked_AMPA)

The receptor layer's job for glutamate is therefore not "how much drive does
glutamate deliver" -- the connectome and the circuit layer already answered
that -- but "what FRACTION of the glutamatergic transmission already in
`W[i,j]` do the drugs on board remove". `share_NMDA + share_AMPA == 1`
partitions the existing weight between the two receptor components; it does
not add to it. With no drug on board every blocked fraction is zero, every
factor is exactly 1, and `W_eff == W`: this module changes nothing about any
result that predates it, which is asserted by a test.

Three consequences worth stating, because they are the honest cost of the
resolution rather than a free lunch:

1. **This module introduces no illustrative coupling gain.** Every other P5
   module has one hand-set `mechanism_gain` per receptor, unidentifiable
   from anything published (P2.3). Here the gain is pinned by the algebra
   above at exactly 1.0 (`GLUTAMATE_CONNECTION_GAIN`) and is not a
   parameter: `1 + gain * activation` must equal `1 - fraction removed`, so
   `gain == 1` and `activation` is the NEGATIVE of the fraction removed.
   The magnitude of glutamatergic transmission was never this module's to
   set. It is `W[i,j]`.

2. **Exogenous glutamate cannot be an excitatory drug in this model.**
   Dosing the `glutamate` ligand raises occupancy at both sites and displaces
   a competitive antagonist, which is real and is tested; what it cannot do
   is increase transmission above `W[i,j]`, because `W[i,j]` already
   represents transmission with the endogenous transmitter present. Making
   it do so would re-introduce exactly the double count. Endogenous
   transmitter tone as a first-class quantity is P6.2, and until it exists
   the model's glutamate ligand is a competitor at the binding site and
   nothing more.

3. **`baseline_input` is not modulated either, and that is a known
   under-statement.** Each node's `baseline_input` is a tonic term standing
   in for everything outside the modelled loop, and part of what it stands
   in for is extrinsic glutamatergic afferents. A systemic NMDA antagonist
   would reduce those too. This module deliberately does not touch
   `baseline_input`, because the term is of mixed and unlabelled origin and
   scaling it would assert a glutamatergic fraction of it that nothing
   measures. The consequence is that this module UNDER-states a systemic
   glutamate antagonist's effect on the circuit, by an amount equal to
   whatever fraction of the tonic drive is glutamatergic. Said once here and
   asserted nowhere, because there is no number to assert.

The membrane potential this model does not have
===============================================

An NMDA receptor's conductance is blocked by extracellular Mg2+ in a
voltage-dependent way ([Nowak1984], [Mayer1984]); the block is relieved as
the postsynaptic membrane depolarizes, which is what makes the receptor a
coincidence detector and what makes NMDA-mediated transmission a nonlinear
function of the postsynaptic state rather than a fixed conductance.

This codebase's circuit layer is a population-RATE model. It has no membrane
potential, in any node, at any time. There is therefore no way to make the
Mg block dynamic here without first inventing a rate-to-voltage map, and no
measurement of "the mean membrane potential of a cortical population firing
at half its maximum rate" was reached (see "Searched and empty").

What this module does instead, and says plainly:

  * The Mg block itself is MEASURED and is implemented exactly, as
    `mg_unblocked_fraction(V)`, from [ChiuCarter2022]'s conductance-voltage
    fits -- mouse, acute cortical slice, 34.7 C, three physiological ionic
    conditions, junction-potential corrected, and fitted over -78 to +32 mV.
    Nothing about that function is hand-set.
  * The VOLTAGE at which it is evaluated is one explicitly ILLUSTRATIVE
    circuit-layer constant, `ILLUSTRATIVE_OPERATING_VOLTAGE`, and every
    headline result below is asserted across the whole measured voltage
    range rather than at that one point. Where a result depends on the
    operating voltage, the dependence is the finding and is reported as one.

The first thing that falls out of doing it exactly is a constraint on the
whole module: at the RESTING membrane potential [ChiuCarter2022] measured in
the same cells (-82.7 mV in their "awake" ionic condition), the Mg block
removes about 97% of the NMDA conductance. So at rest the excitatory
connectivity weight is essentially pure AMPA, and an NMDA antagonist can
have essentially no circuit effect. NMDA's share of `W[i,j]` -- and hence
ketamine's leverage on this circuit -- is a monotonically increasing
function of how depolarized the postsynaptic population is. A test asserts
that, because it is the single most load-bearing prediction in the module.

Sources
-------

Agonist affinities. ONE laboratory, ONE preparation, BOTH receptor types,
which is what makes the NMDA-versus-AMPA contrast below a comparison rather
than two quotations, and MOUSE, which is this project's species:

  [Patneau1990] Patneau DK, Mayer ML. "Structure-activity relationships for
                amino acid transmitter candidates acting at N-methyl-D-
                aspartate and quisqualate receptors." J Neurosci.
                1990;10(7):2385-2399. (PMID:2165523, PMC6570388.) Whole-cell
                voltage clamp of MOUSE embryonic hippocampal neurons in
                culture. L-glutamate EC50:
                  NMDA receptors                  2.3 uM
                    (3 uM glycine, no extracellular Mg2+)
                  quisqualate/AMPA receptors,
                    rapidly desensitizing        480 uM
                    non-desensitizing             19 uM
                    (no added glycine, 1 mM extracellular Mg2+)
                The 25-fold gap between the two AMPA-receptor numbers is
                that paper's own, from one preparation, and is carried
                below rather than collapsed: they are two response
                components of the same receptor population, and this model
                has one occupancy state for it. The paper also reports the
                limiting log-log slope of NMDA receptor current versus
                agonist concentration as 1.93, i.e. TWO agonist binding
                sites; `ReceptorSite` is one-site, which is recorded under
                "What was deliberately not built".

NMDA channel block. Two laboratories, and they disagree by 10-fold in a
direction the mechanism predicts:

  [Dravid2007]  Dravid SM, Erreger K, Yuan H, Nicholson K, Le P,
                Lyuboslavsky P, Almonte A, Murray E, Mosely C, Barber J,
                French A, Balster R, Murray TF, Traynelis SF.
                "Subunit-specific mechanisms and proton sensitivity of NMDA
                receptor channel block." J Physiol. 2007;581(Pt 1):107-128.
                doi:10.1113/jphysiol.2006.124958 (PMID:17303642,
                PMC2075223). RAT cDNAs in Xenopus oocytes, two-electrode
                voltage clamp, holding potential -40 mV, 23 C, 50 uM
                glutamate + 30 uM glycine, bath (mM) 90 NaCl / 1 KCl /
                10 HEPES / 0.5 BaCl2 -- i.e. NO Mg2+ and no Ca2+. Table 2
                IC50 at pH 7.6, 6-42 oocytes per point:
                  (+)MK-801  NR2A 0.015  NR2B 0.009  NR2C 0.024  NR2D 0.038 uM
                  ketamine   NR2A 3.31   NR2B 0.927  NR2C 1.65   NR2D 2.42  uM
                and Table 5, MK-801 rate constants at NR1/NR2A from
                macroscopic response relaxation, pH 7.6:
                  (+)MK-801  k_ON 3.9e6 M^-1 s^-1  k_OFF 0.12 s^-1  Kd 33 nM
                  (-)MK-801  k_ON 0.36e6           k_OFF 0.15       Kd 416 nM
                The (+)MK-801 pair is the ONLY measured association rate
                constant reached for any ligand in this module, and it is
                what lets the NMDA channel-blocker pool use derivation 1
                rather than an illustrative `kon`.
  [Kotermanski2009] Kotermanski SE, Johnson JW. "Mg2+ imparts NMDA receptor
                subtype selectivity to the Alzheimer's drug memantine."
                J Neurosci. 2009;29(9):2774-2779.
                doi:10.1523/JNEUROSCI.3703-08.2009 (PMID:19261873,
                PMC2679254). RAT cDNAs in HEK293T cells, whole cell, holding
                potential -66 mV, 1 mM glutamate + 100 uM glycine, n=3-4
                cells per condition. Ketamine IC50:
                                   0 Mg2+      1 mM Mg2+
                  GluN1/2A         0.33 uM      5.35 uM
                  GluN1/2B         0.51 uM      5.08 uM
                  GluN1/2C         1.18 uM      1.18 uM
                  GluN1/2D         2.95 uM      2.95 uM
                (memantine: 0.80/0.52/1.61/1.76 uM in 0 Mg2+ and
                13.4/10.4/1.61/1.76 uM in 1 mM Mg2+). 1 mM Mg2+ raises the
                GluN2A/GluN2B IC50s ~17-fold because Mg2+ and the drug
                compete for the same site, and only ~3-fold at GluN2C/2D
                where Mg2+ is less potent. This is the paper that makes
                "which ketamine constant" a question with a real answer;
                see "Which ketamine affinity, and why it is not the tightest
                one".
  [Huettner1988] Huettner JE, Bean BP. "Block of N-methyl-D-aspartate-
                activated current by the anticonvulsant MK-801: selective
                binding to open channels." Proc Natl Acad Sci USA.
                1988;85(4):1307-1311. doi:10.1073/pnas.85.4.1307
                (PMID:2448800, PMC279756). RAT neocortical neurons in cell
                culture. The MECHANISM -- "binding and unbinding of MK-801
                seems to be possible only if the N-Me-D-Asp-operated channel
                is in the transmitter-activated state", block is
                open-channel and Mg2+ inhibits it at negative potentials --
                and the one number this module uses as a falsifier rather
                than as an input: recovery from block during continuous
                agonist has a time constant of ~90 min at -70 to -80 mV and
                ~2 min at +30 mV. See "The MK-801 unblocking rate this model
                gets wrong by 650-fold".
  [Wong1986]    Wong EHF, Kemp JA, Priestley T, Knight AR, Woodruff GN,
                Iversen LL. "The anticonvulsant MK-801 is a potent
                N-methyl-D-aspartate antagonist." Proc Natl Acad Sci USA.
                1986;83(18):7104-7108. doi:10.1073/pnas.83.18.7104
                (PMID:3529096, PMC386661). [3H]MK-801 binding to RAT brain
                membranes, Kd = 37.2 +/- 2.7 nM. An equilibrium BINDING
                constant in native tissue, recorded as the independent check
                on [Dravid2007]'s kinetically derived 33 nM; they agree to
                13%, across two laboratories, two preparations and two
                methods, which is the tightest agreement anywhere in this
                module.

AMPA competitive antagonism:

  [Randle1992]  Randle JCR, Guet T, Cordi A, Lepagnol JM. "Competitive
                inhibition by NBQX of kainate/AMPA receptor currents and
                excitatory synaptic potentials: importance of 6-nitro
                substitution." Eur J Pharmacol. 1992;215(2-3):237-244.
                doi:10.1016/0014-2999(92)90033-Z (PMID:1382998). Two-
                electrode voltage clamp of Xenopus oocytes injected with RAT
                cortex mRNA. NBQX apparent Ki 63 nM versus AMPA and 78 nM
                versus kainate; Schild analysis gave pA2 7.05 +/- 0.10
                (versus AMPA) and 7.17 +/- 0.05 (versus kainate) with SLOPES
                NEAR UNITY, "confirming the competitive nature of the
                inhibition". That last clause is why NBQX sits in the
                orthosteric pool and ketamine and MK-801 do not: for NBQX
                the competitive mechanism is the measurement, not an
                assumption. The same paper reports NBQX IC50 of 0.90 uM
                against kainate/AMPA-mediated field EPSPs in rat hippocampal
                CA1 slices, and describes the "10-fold differences between
                the IC50 values in hippocampal slices and the Ki values in
                Xenopus oocytes" as correlating "closely with data for other
                quinoxalinedione derivative antagonists". On the two AMPA
                numbers specifically that offset is 14-fold, which is what a
                test asserts.
  [Sheardown1990] Sheardown MJ, Nielsen EO, Hansen AJ, Jacobsen P, Honore T.
                "2,3-Dihydroxy-6-nitro-7-sulfamoyl-benzo(F)quinoxaline: a
                neuroprotectant for cerebral ischemia." Science.
                1990;247(4942):571-574. doi:10.1126/science.2154034
                (PMID:2154034). The paper NBQX comes from, cited for
                selectivity -- "a potent and selective inhibitor of binding
                to the quisqualate subtype of the glutamate receptor, with
                no activity at the NMDA and glycine sites" -- which is what
                licenses this module to give NBQX a ligand entry at the AMPA
                site and none at the NMDA site. Its widely quoted [3H]AMPA
                binding IC50 of 0.15 uM is NOT used here: the number is not
                in the abstract, the full text was not retrievable, and
                [Randle1992]'s functional Ki is both retrievable and the
                right kind of constant.

The magnesium block and the NMDA share of a synapse:

  [ChiuCarter2022] Chiu DN, Carter BC. "Synaptic NMDA receptor activity at
                resting membrane potentials." Front Cell Neurosci.
                2022;16:916626. doi:10.3389/fncel.2022.916626. MOUSE (CD1,
                P11-24), acute slices, layer 2/3 pyramidal neurons of
                primary somatosensory cortex and CA1 pyramidal neurons,
                34.7 +/- 0.9 C, junction-potential corrected. Normalized
                NMDA-receptor conductance-voltage fits, over -78 to +32 mV:
                  "classic" ACSF (1.0 mM Mg2+, 2.0 Ca2+, 2.5 K+)
                      f(V) = 1/[1 + 0.11  exp(-0.074 V)]   midpoint -29.5 mV
                  "asleep" ACSF  (0.8 mM Mg2+, 1.3 Ca2+, 3.8 K+)
                      f(V) = 1/[1 + 0.095 exp(-0.071 V)]   midpoint -33.1 mV
                  "awake" ACSF   (0.7 mM Mg2+, 1.2 Ca2+, 4.2 K+)
                      f(V) = 1/[1 + 0.073 exp(-0.074 V)]   midpoint -35.6 mV
                (n = 11 cortical recordings for classic and awake, 7 for
                asleep.) Resting membrane potential of the same L2/3
                pyramidal neurons, n = 27: -89.4 +/- 0.9 mV (classic),
                -84.5 +/- 0.9 (asleep), -82.7 +/- 1.0 (awake). NMDA
                reversal potential 0.02 +/- 1.00 mV (n = 11).
  [JahrStevens1990] Jahr CE, Stevens CF. "Voltage dependence of
                NMDA-activated macroscopic conductances predicted by
                single-channel kinetics." J Neurosci. 1990;10(9):3178-3182.
                (PMID:1697902, PMC6570236.) Cultured hippocampal neurons,
                whole cell, extracellular Mg2+ from 1 uM to 10 mM. The
                canonical form of the same function, universally quoted as
                `1/(1 + [Mg2+]/3.57 * exp(-0.062 V))` with [Mg2+] in mM and
                V in mV. RECORDED, NOT USED, and the reason is stated: the
                full text sits behind a publisher block that returned HTTP
                403 to every retrieval attempt, so those two constants could
                not be checked against the paper that is supposed to contain
                them, and this module does not put unverifiable numbers in
                its equations when a verifiable, same-species, physiological-
                temperature alternative is in hand. `JAHR_STEVENS_1990_*`
                below carry the quoted values so the comparison is
                available, and a test compares the two functions rather than
                letting the difference go unexamined.
  [Myme2003]    Myme CI, Sugino K, Turrigiano GG, Nelson SB. "The
                NMDA-to-AMPA ratio at synapses onto layer 2/3 pyramidal
                neurons is conserved across prefrontal and visual cortices."
                J Neurophysiol. 2003;90(2):771-779.
                doi:10.1152/jn.00070.2003 (PMID:12672778). RAT, P16-21,
                layer 2/3 pyramidal cells of medial prefrontal and visual
                cortex, whole-cell voltage clamp in brain slices. Mixed
                miniature EPSCs with AMPA- and NMDA-mediated components,
                isolated in NOMINALLY 0 Mg2+ ACSF, averaged and fitted with
                double exponentials: NMDA/AMPA peak ratio 27 +/- 1% in PFC
                (n = 27) and 28 +/- 3% in visual cortex (n = 28), NMDA decay
                69 +/- 8 ms and 67 +/- 6 ms. The 0 Mg2+ condition is exactly
                what this model needs and is the reason this source was
                chosen: it is the NMDA:AMPA conductance ratio with the Mg2+
                block REMOVED, so it composes with [ChiuCarter2022]'s block
                function without either being counted twice. The same paper
                notes that EPSCs evoked in 2 mM Mg2+ at depolarized
                potentials gave a LARGER average ratio, and that is recorded
                below as the upper edge of the range.
  [Bogart2018]  Bogart LJ, O'Donnell P. "Multiple long-range inputs evoke
                NMDA currents in prefrontal cortex fast-spiking
                interneurons." Neuropsychopharmacology. 2018;43(10):2101-
                2108. doi:10.1038/s41386-018-0029-5 (PMID:29483660,
                PMC6098111). MOUSE (GAD1-EGFP), adult, mPFC slices, 1 mM
                Mg2+, holding -55 mV, picrotoxin, NMDA component isolated as
                the NBQX-resistant current. NMDAR:AMPAR peak ratio, fast-
                spiking interneuron versus pyramidal cell, per input:
                  contralateral mPFC   FSI 0.62 +/- 0.11   PC 5.12 +/- 1.70
                  ventral hippocampus  FSI 0.26 +/- 0.05   PC 2.01 +/- 0.88
                  mediodorsal thalamus FSI 0.54 +/- 0.13   PC 5.44 +/- 1.41
                i.e. the interneuron ratio is 8-10 fold SMALLER. Used for
                its RELATIVE statement only, one laboratory and one
                preparation, and used to REFUSE a modelling shortcut rather
                than to license one; see "Why the interneuron is not given
                more NMDA receptor".

Circuit-level target:

  [Homayoun2007] Homayoun H, Moghaddam B. "NMDA receptor hypofunction
                produces opposite effects on prefrontal cortex interneurons
                and pyramidal neurons." J Neurosci. 2007;27(43):11496-11500.
                doi:10.1523/JNEUROSCI.2213-07.2007 (PMID:17959792,
                PMC2954603). Single-unit recording in AWAKE, freely moving
                RATS, MK-801 0.1 mg/kg i.p. "~69% of FS neurons decreased
                their firing rate", while "86% of RS, recorded during the
                same sessions, increased their firing rate" (chi2 = 100.42,
                p < 0.001), and interneuron inhibition preceded pyramidal
                excitation. The authors' conclusion is the one this module
                is tested against: "NMDA receptors preferentially drive the
                activity of cortical inhibitory interneurons suggesting that
                NMDA receptor inhibition causes cortical excitation by
                disinhibition of pyramidal neurons." This is a DIRECTION,
                not a dose-response curve, and it is used as one: no
                parameter here is fitted to it, and the test that checks it
                checks signs.
  [Nowak1984]   Nowak L, Bregestovski P, Ascher P, Herbet A, Prochiantz A.
                "Magnesium gates glutamate-activated channels in mouse
                central neurones." Nature. 1984;307(5950):462-465.
                doi:10.1038/307462a0 (PMID:6320006).
  [Mayer1984]   Mayer ML, Westbrook GL, Guthrie PB. "Voltage-dependent block
                by Mg2+ of NMDA responses in spinal cord neurones." Nature.
                1984;309(5965):261-263. doi:10.1038/309261a0
                (PMID:6325946). The two papers that established the
                voltage-dependent Mg2+ block. No number is taken from
                either; the mechanism is.

Derivations
-----------

P5.1's three-way split, with P5.4's fourth case, and this module is the
first to reach case 1:

1. **Both binding constants measured** -- use them. (+)MK-801 at the NMDA
   channel site: `k_ON = 3.9e6 M^-1 s^-1` and `k_OFF = 0.12 s^-1`
   ([Dravid2007] Table 5), converted to this codebase's units by
   `kon_per_molar_second_to_per_nM_min` and `per_second_to_per_min`. Their
   ratio reproduces that table's own printed Kd of 33 nM and agrees within
   13% with [Wong1986]'s independent binding Kd of 37.2 nM; a test asserts
   both.
2. **koff and an equilibrium affinity measured, kon missing** -- not needed
   here.
3. **Only an equilibrium affinity measured** -- share the SITE's (or, here,
   the POOL's) measured `kon` and set `koff = kon * Kd`. Ketamine takes
   (+)MK-801's measured `kon`, because they are the same mechanism at the
   same site in the same channel lumen, and its own `kon` was not measured
   anywhere reached. Glutamate and NBQX have no measured `kon` at either
   orthosteric site, so those pools share `ILLUSTRATIVE_KON`, the same
   explicitly hand-set 0.01 nM^-1 min^-1 as `acetylcholine.ILLUSTRATIVE_KON`
   (a test asserts the equality). Every affinity is reproduced EXACTLY by
   construction; only the approach timescale is illustrative.
4. **A potency, not an affinity.** Every glutamate and ketamine number here
   is a functional EC50 or IC50 from voltage clamp, not a binding Kd, and is
   used as the site's Kd. Same assumption as P5.2 to P5.4: activation
   proportional to occupancy with no receptor reserve, which is what
   `activation = sum(occupancy * efficacy)` already asserts.

Which ketamine affinity, and why it is not the tightest one
-----------------------------------------------------------

Three measured ketamine potencies at a GluN2A-containing NMDA receptor are
in hand and they span 16-fold:

    0.33 uM  [Kotermanski2009]  HEK293T, -66 mV, 0 Mg2+
    3.31 uM  [Dravid2007]       oocyte,  -40 mV, 0 Mg2+ (0.5 mM Ba2+)
    5.35 uM  [Kotermanski2009]  HEK293T, -66 mV, 1 mM Mg2+

The two Mg2+-free values differ 10-fold between laboratories, and the
difference runs in the direction the mechanism requires: open-channel block
is voltage dependent and deepens with hyperpolarization, and the tighter
value is the one measured 26 mV more negative. That agreement of sign is why
neither is discarded.

`NMDA_KD_KETAMINE` is the 1 mM Mg2+ value, 5.35 uM, and the reason is
specific rather than conservative: Mg2+ and ketamine compete for the same
site, so in a brain -- which has Mg2+ -- ketamine's effective potency is the
Mg2+-containing one. The stated cost is that the model then carries ONE
ketamine potency, measured at ONE voltage (-66 mV) in ONE Mg2+
concentration, while `ILLUSTRATIVE_OPERATING_VOLTAGE` is free to move.
Ketamine's potency does not track the model's operating voltage; the Mg2+
block of the CONDUCTANCE does. That is an approximation, it is in the
conservative direction as the operating voltage depolarizes (real block
would weaken further, so the model over-states ketamine at depolarized
potentials), and it is not hidden behind an average.

The MK-801 unblocking rate this model gets wrong by 650-fold
------------------------------------------------------------

[Dravid2007]'s `k_OFF` of 0.12 s^-1 is a 5.8-second unblocking time
constant. [Huettner1988] measured recovery from MK-801 block, in continuous
agonist, at ~90 MINUTES at -70 to -80 mV, and ~2 minutes at +30 mV. Those
are the same molecule at the same site and they differ by roughly 650-fold
at physiological potentials.

Both are right about different things. [Dravid2007]'s number is a relaxation
rate fitted at -40 mV in a Mg2+-free, Ba2+-substituted bath from a kinetic
scheme in which the blocker leaves an open channel; [Huettner1988]'s is the
recovery of the macroscopic current, which for a TRAPPING blocker requires
the channel to reopen with the blocker still inside and is therefore limited
by channel gating and strongly voltage dependent. MK-801's clinical and
behavioural signature -- effectively irreversible block after a single dose
-- is [Huettner1988]'s number, not [Dravid2007]'s.

This module uses [Dravid2007]'s pair because it needs a `kon`/`koff` pair
from one experiment and that is the only one reached, and it states here
that the resulting MK-801 washout is therefore something like 650-fold too
fast at resting potential. A test asserts the ratio so the claim cannot rot.
Nothing in this module's headline results depends on MK-801's washout: the
circuit result is ketamine's, and the MK-801 result is about the agreement
of its equilibrium constants across three sources.

Why the interneuron is not given more NMDA receptor
----------------------------------------------------

[Homayoun2007]'s explanation for their own result is that "NMDA receptors
preferentially drive the activity of cortical inhibitory interneurons". The
tempting move is to hard-code that: give the pyramidal-to-interneuron
connection a larger NMDA share than any other, and the disinhibition falls
out by construction.

That move is refused, for two reasons and one of them is a measurement.
[Bogart2018] measured the NMDAR:AMPAR ratio at excitatory synapses onto
prefrontal fast-spiking interneurons and onto pyramidal cells, in one
laboratory, one preparation, three separate long-range inputs, and found the
interneuron ratio 8 to 10-fold SMALLER, not larger. Whatever makes NMDA
antagonists preferentially silence interneurons in vivo, the postsynaptic
NMDA:AMPA ratio is not it -- and a receptor-layer parameter set to
contradict a direct receptor-layer measurement, in order to reproduce a
circuit-layer result, would be exactly the substitution this project's rule
2 forbids.

The second reason is that it is unnecessary. In the cortical E-I loop this
module modulates, the ONLY glutamatergic connection is the pyramidal ->
interneuron one; the return connection is GABAergic. So reducing
glutamatergic transmission necessarily reduces the interneuron's drive
first, and the pyramidal population is disinhibited as a consequence of the
loop's topology rather than of a tuned receptor parameter. Both nodes get
the SAME NMDA share, from the same two measurements, and the result is a
statement about the connectivity matrix.

What that buys, and what it costs, is stated in `docs/architecture.md`, P5.5:
the result is unconditional in a two-population loop with no recurrent
excitation, and conditional once recurrent pyramidal-to-pyramidal excitation
is added, with the threshold reported. A test builds the recurrent variant
and measures where the direction flips rather than asserting it cannot.

Searched and empty, or found and deliberately not used
------------------------------------------------------

* **Association rate constants for glutamate, ketamine or NBQX at any of
  these receptors.** Searched Europe PMC for each ligand crossed with
  "association rate constant", "binding kinetics", "kon" and "residence
  time", and through the NMDA/AMPA channel-block kinetics literature.
  [Dravid2007] Table 5 is the only measured `k_ON` reached and it is for the
  MK-801 stereoisomers only; that paper's own ketamine analysis is an IC50
  and a pH dependence, not a rate pair. Glutamate's binding rate at either
  ionotropic receptor is fast enough that the published kinetics are about
  channel gating rather than about binding, and no `kon` for it was reached.
  Hence `ILLUSTRATIVE_KON` for the two orthosteric pools.
* **The mean membrane potential of a cortical population firing at half its
  maximum rate.** This is the quantity `ILLUSTRATIVE_OPERATING_VOLTAGE`
  stands in for and it is not a quantity anyone measures, because
  "population firing at half its maximum rate" is a property of this model
  and not of a preparation. Searched for measured in-vivo up-state and
  down-state membrane potentials and for action-potential thresholds of
  mouse cortical pyramidal neurons that could anchor the depolarized end;
  [Steriade1993] (Steriade M, Nunez A, Amzica F, J Neurosci 1993;13:3252-
  3265, PMID:8340806) describes the depolarizing envelope qualitatively and
  gives the potentials at which it is SUPPRESSED (-90 to -100 mV) rather
  than the potentials it reaches, and the action-potential-threshold values
  reachable were either from a disease-model preparation or from database
  aggregates without a primary measurement behind the specific number. The
  RESTING end is measured ([ChiuCarter2022], -82.7 to -89.4 mV in three
  ionic conditions) and is used as the anchor for the one thing that end
  can settle: that at rest the NMDA share is negligible. The depolarized
  end stays illustrative and every result is swept across it.
* **A behavioural dose-response usable as a circuit-layer target.** Not
  searched again. [Homayoun2007]'s 0.1 mg/kg i.p. MK-801 is an
  ADMINISTERED DOSE, which is exactly what Phase 1 built the bridge for, and
  it is exactly what this module cannot use, because no rodent PK parameter
  set (F, ka, Vd, CL, brain partition) for MK-801 or ketamine has been
  ingested into `drug_pk.py`. That is a real, specific, buildable follow-up
  and it is named here rather than left implicit: with an MK-801 PK set,
  [Homayoun2007] becomes a quantitative target instead of a sign test.
* **Rodent pharmacokinetics for ketamine, MK-801 and NBQX.** All three have
  published rodent PK. Not used, for P5.3's and P5.4's two reasons,
  unchanged: `drug_pk.py` consumes a full per-route parameter set rather
  than a half-life, and half-sourcing one of the three would put a real
  citation beside two invented numbers and assert a clearance ORDERING
  nothing supports. All three share one illustrative `TOOL_COMPOUND_KE`,
  deliberately the same number as `serotonin.TOOL_COMPOUND_KE`,
  `dopamine.TOOL_COMPOUND_KE`, `noradrenaline.TOOL_COMPOUND_KE` and
  `acetylcholine.TOOL_COMPOUND_KE`; a test asserts the equality.
* **Glutamate's own clearance rate.** The same gap the four transmitters
  before it have. `GLUTAMATE_KE` is an explicit flagged proxy set to the
  same 5.0 min^-1 as `dopamine.DOPAMINE_KE`, `serotonin.SEROTONIN_KE`,
  `noradrenaline.NORADRENALINE_KE` and `acetylcholine.ACETYLCHOLINE_KE`.
  Extracellular glutamate is cleared by EAAT transporters on a millisecond
  timescale in the cleft; the real quantity belongs to P6.2.

What was deliberately not built
-------------------------------

* **The glycine/D-serine co-agonist site.** An NMDA receptor needs BOTH
  glutamate and a co-agonist to open, which is why [Patneau1990] adds 3 uM
  glycine and [Dravid2007] adds 30 uM. This module has one orthosteric pool
  per site and therefore cannot represent an obligatory two-agonist AND
  gate; every NMDA constant used here was measured with the co-agonist site
  saturated, so the model is the saturated-co-agonist limit and says so.
  Anything about D-serine, glycine transport, or co-agonist-site partial
  agonism is out of reach until the site gains a second pool.
* **Two agonist binding sites per NMDA receptor.** [Patneau1990]'s limiting
  log-log slope of 1.93 says the NMDA receptor binds two glutamate
  molecules; `ReceptorSite` is a one-site, Hill-coefficient-1 model. The
  consequence is that this module's NMDA occupancy curve is shallower than
  the measured one away from the EC50. Nothing here depends on the slope,
  because the NMDA result is driven by the uncompetitive blockers, whose
  occupancy is genuinely one-site.
* **AMPA receptor desensitization.** [Patneau1990] measures glutamate's
  potency at TWO components of the AMPA-receptor response, a rapidly
  desensitizing one (EC50 480 uM) and a non-desensitizing one (19 uM), and
  P5.4 built `DesensitizingReceptorSite` for exactly this shape of problem.
  It is not used here, on purpose: what P5.4's class represents is an
  agonist-driven, slowly-recovering withdrawal of receptor from the
  available pool, with rate constants measured in minutes; AMPA receptor
  desensitization recovers in tens of milliseconds and its role at a
  synapse is to shape a single EPSC, three to four orders of magnitude
  below any timescale this codebase integrates. Both EC50s are recorded and
  the desensitizing one is instantiated, because it is the component that
  carries the synaptic current.
* **Use- and voltage-dependence of ketamine's and MK-801's block.** Same
  omission, same reason, as P5.4's mecamylamine: this engine has no
  membrane potential and no open-channel state, so "block only accumulates
  while the channel is open" cannot be written down. The multiplicative
  form reproduces what a functional IC50 measures at one holding potential.
  What follows is that block in this model does not grow with agonist
  concentration, and real open-channel block does -- which for ketamine and
  MK-801, whose selectivity for active circuits is the whole story, is the
  most consequential simplification in the module.
* **NMDA receptor subtype composition per region.** [Dravid2007] and
  [Kotermanski2009] both resolve GluN2A-D and the potencies span 4-fold
  (MK-801) and 9-fold (ketamine, 0 Mg2+); all of it is recorded in
  `NMDA_SUBTYPE_*` below and none of it is instantiated, because this
  two-node loop has no node whose identity would fix its GluN2 composition.
  `Grin1`, `Grin2a`, `Grin2b`, `Gria1` to `Gria4` are already tagged P5.5 in
  `expression/genes.py`, so the regional version is a table lookup away and
  is Phase 4's to spend, not this task's.
* **An atlas-derived density factor.** Same position as P5.1 to P5.4, and a
  sharper one here: a receptor DENSITY does not enter this module's
  arithmetic at all, because the quantity being modulated is a SHARE of an
  existing weight and shares sum to one. Regional receptor density would
  change the NMDA:AMPA share per region, which is a genuinely different and
  better use of the expression maps than scaling a coupling gain.
* **Kainate receptors.** [Randle1992] shows NBQX is essentially equipotent
  at AMPA and kainate receptors (Ki 63 nM versus 78 nM), so "NBQX blocks
  AMPA receptors" is, at these concentrations, "NBQX blocks AMPA AND
  kainate receptors". The roadmap asks for AMPA; the cross-reactivity is
  recorded as a constant so it is a number rather than a caveat.
"""

import numpy as np

from .behavior import BehaviorReadout
from .circuit import CircuitNode
from .engine import ReceptorSystem
from .network import CircuitNetwork
from .pk import OneCompartmentPK
from .receptor import ChannelBlockingReceptorSite, Ligand, ReceptorSite


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------


def kon_per_molar_second_to_per_nM_min(kon):
    """M^-1 s^-1 -> nM^-1 min^-1.

    One measured association rate constant exists in this module
    ([Dravid2007] Table 5) and it is published in the electrophysiologist's
    units; the conversion is written once here rather than open-coded at the
    constant, so it is checkable. 1 M^-1 s^-1 = 1e-9 nM^-1 * 60 min^-1.
    """
    return kon * 60.0 * 1e-9


def per_second_to_per_min(rate):
    """s^-1 -> min^-1."""
    return rate * 60.0


# ---------------------------------------------------------------------------
# The magnesium block: measured, and the only place a voltage appears
# ---------------------------------------------------------------------------

#: [ChiuCarter2022]'s three measured conductance-voltage fits, as
#: (prefactor, slope_per_mV, midpoint_mV, mg_mM), giving
#: `f(V) = 1 / (1 + prefactor * exp(-slope * V))`. MOUSE, acute cortical
#: slice, 34.7 C, junction-potential corrected, fitted over -78 to +32 mV.
#: The three differ in the whole ionic composition of the bath, not in Mg2+
#: alone, which is why the key is the paper's own condition name.
MG_BLOCK_FITS = {
    "classic": (0.11, 0.074, -29.5, 1.0),
    "asleep": (0.095, 0.071, -33.1, 0.8),
    "awake": (0.073, 0.074, -35.6, 0.7),
}

#: The default ionic condition. "awake" is chosen over the "classic" 1.0 mM
#: Mg2+ recipe because [ChiuCarter2022] built it from measured interstitial
#: ion concentrations in the AWAKE brain, and every behavioural readout this
#: project is aimed at is measured in an awake animal. The classic condition
#: is a 1.4-fold tighter block at -60 mV and is one keyword away; a test
#: reports the spread across all three rather than letting the choice hide.
DEFAULT_IONIC_CONDITION = "awake"

#: [JahrStevens1990]'s canonical constants as universally quoted:
#: `f(V) = 1/(1 + [Mg2+]/3.57 * exp(-0.062 V))`, [Mg2+] in mM. RECORDED, NOT
#: USED -- the primary could not be retrieved to verify them (HTTP 403), and
#: this module prefers the fits it can check. `jahr_stevens_unblocked_
#: fraction` exists so the two can be compared, and a test does compare them.
JAHR_STEVENS_1990_SLOPE_PER_MV = 0.062
JAHR_STEVENS_1990_MG_SCALE_MM = 3.57


def mg_unblocked_fraction(v_mV, condition=DEFAULT_IONIC_CONDITION):
    """Fraction of NMDA-receptor conductance NOT blocked by Mg2+ at `v_mV`.

    MEASURED, [ChiuCarter2022]. Monotonically increasing in voltage, ~0 at
    resting potential and ~1 at 0 mV: this single function is the reason an
    NMDA antagonist's circuit effect in this model depends on how depolarized
    the target population is.
    """
    if condition not in MG_BLOCK_FITS:
        raise ValueError(
            "unknown ionic condition %r; expected one of %s"
            % (condition, sorted(MG_BLOCK_FITS))
        )
    prefactor, slope, _midpoint, _mg = MG_BLOCK_FITS[condition]
    v = np.asarray(v_mV, dtype=float)
    return 1.0 / (1.0 + prefactor * np.exp(-slope * v))


def jahr_stevens_unblocked_fraction(v_mV, mg_mM):
    """The same quantity in [JahrStevens1990]'s canonical form.

    Provided for comparison only; nothing in this module calls it. See the
    note on `JAHR_STEVENS_1990_SLOPE_PER_MV` for why it is not the default.
    """
    v = np.asarray(v_mV, dtype=float)
    scaled = mg_mM / JAHR_STEVENS_1990_MG_SCALE_MM
    return 1.0 / (1.0 + scaled * np.exp(-JAHR_STEVENS_1990_SLOPE_PER_MV * v))


#: mV. MEASURED resting membrane potentials of the L2/3 pyramidal neurons
#: [ChiuCarter2022] fitted the block in (n = 27), one per ionic condition.
#: These are what anchors the statement that the NMDA share is negligible at
#: rest; they are NOT the model's operating point (see below).
RESTING_POTENTIAL_MV = {"classic": -89.4, "asleep": -84.5, "awake": -82.7}

#: mV. ILLUSTRATIVE, and the only free parameter in this module's receptor
#: arithmetic. The single postsynaptic membrane potential at which the Mg2+
#: block is evaluated, standing in for a quantity a rate model does not have
#: (see "The membrane potential this model does not have"). -40 mV is a
#: round value in the depolarized half of [ChiuCarter2022]'s measured -78 to
#: +32 mV range, i.e. inside the data rather than extrapolated from it, and
#: well depolarized from the measured resting potential because this
#: codebase's nodes sit at half their maximum firing rate at baseline. It is
#: NOT a measured membrane potential of anything. Every result in this module
#: is swept across the measured voltage range instead of being asserted here.
ILLUSTRATIVE_OPERATING_VOLTAGE = -40.0


# ---------------------------------------------------------------------------
# The NMDA share of an excitatory connectivity weight
# ---------------------------------------------------------------------------

#: MEASURED NMDA:AMPA peak-current ratio of mixed mEPSCs in NOMINALLY ZERO
#: Mg2+, i.e. with the block relieved: 27 +/- 1% at rat medial prefrontal
#: layer 2/3 pyramidal cells (n = 27) and 28 +/- 3% at visual cortex
#: (n = 28) [Myme2003]. The 0 Mg2+ condition is why this number composes
#: with `mg_unblocked_fraction` without double-counting the block.
MYME2003_NMDA_AMPA_RATIO_PFC = 0.27
MYME2003_NMDA_AMPA_RATIO_VISUAL = 0.28

#: The ratio this module uses: the prefrontal value, because the circuit
#: below is a prefrontal one.
NMDA_AMPA_CONDUCTANCE_RATIO = MYME2003_NMDA_AMPA_RATIO_PFC

#: A PEAK-current ratio understates NMDA's share of transmitted CHARGE,
#: because [Myme2003] measured the NMDA component's decay at 69 +/- 8 ms
#: while an AMPA EPSC decays in a few ms. A rate model integrates charge, not
#: peaks. The direction of the error is therefore known -- this module
#: under-states the NMDA share -- and its size is not, because the matching
#: AMPA decay constant is not in the abstract-level record that could be
#: retrieved. [Myme2003] also reports that EPSCs evoked in 2 mM Mg2+ at
#: depolarized potentials gave a LARGER average ratio than the mEPSC value,
#: without stating it numerically in the record retrieved. Both point the
#: same way, so `NMDA_AMPA_RATIO_IS_A_LOWER_BOUND` is asserted by a test
#: rather than corrected by a guess.
NMDA_AMPA_RATIO_IS_A_LOWER_BOUND = True

#: MEASURED NMDAR:AMPAR ratios at fast-spiking interneurons versus pyramidal
#: cells, one laboratory, three inputs [Bogart2018], as
#: input -> (FSI, pyramidal). Recorded and NOT instantiated: see "Why the
#: interneuron is not given more NMDA receptor". A test asserts that the
#: model gives both nodes the same share despite this table existing.
BOGART2018_FSI_VS_PYRAMIDAL_RATIO = {
    "contralateral_mPFC": (0.62, 5.12),
    "ventral_hippocampus": (0.26, 2.01),
    "mediodorsal_thalamus": (0.54, 5.44),
}


def nmda_share(
    v_mV=ILLUSTRATIVE_OPERATING_VOLTAGE,
    ratio=NMDA_AMPA_CONDUCTANCE_RATIO,
    condition=DEFAULT_IONIC_CONDITION,
):
    """Fraction of an excitatory connectivity weight carried by NMDA receptors.

    Composes the two measurements: `ratio` is the NMDA:AMPA conductance ratio
    with the Mg2+ block removed ([Myme2003], 0 Mg2+), and
    `mg_unblocked_fraction` is how much of the NMDA conductance survives the
    block at `v_mV` ([ChiuCarter2022], physiological Mg2+). With
    `g_N = ratio * b(V)` and `g_A = 1` in AMPA units,

        share_NMDA = g_N / (g_N + g_A)

    so the two shares sum to 1 by construction and partition the weight
    rather than adding to it. That identity is the whole double-counting
    resolution in one line, and a test asserts it.
    """
    g_nmda = ratio * mg_unblocked_fraction(v_mV, condition=condition)
    return g_nmda / (1.0 + g_nmda)


def ampa_share(
    v_mV=ILLUSTRATIVE_OPERATING_VOLTAGE,
    ratio=NMDA_AMPA_CONDUCTANCE_RATIO,
    condition=DEFAULT_IONIC_CONDITION,
):
    """The complement of `nmda_share`: the AMPA fraction of the same weight."""
    return 1.0 - nmda_share(v_mV, ratio=ratio, condition=condition)


# ---------------------------------------------------------------------------
# Binding constants
# ---------------------------------------------------------------------------

#: nM. MEASURED L-glutamate EC50 at NMDA receptors on mouse embryonic
#: hippocampal neurons, 2.3 uM, with the glycine co-agonist site saturated
#: (3 uM glycine) and no extracellular Mg2+ [Patneau1990]. Derivation 4.
NMDA_KD_GLUTAMATE = 2300.0

#: nM. MEASURED L-glutamate EC50 at quisqualate/AMPA receptors in the SAME
#: paper and the SAME preparation, for the rapidly desensitizing response
#: (480 uM) and for the small-amplitude non-desensitizing response (19 uM)
#: [Patneau1990]. The desensitizing value is the one instantiated: it is the
#: component that carries the synaptic current. Both are kept because the
#: 25-fold gap between them is the honest error bar on any single
#: "glutamate's affinity for the AMPA receptor" statement, and a test asserts
#: the ratio.
AMPA_KD_GLUTAMATE = 480000.0
AMPA_KD_GLUTAMATE_NONDESENSITIZING = 19000.0

#: nM. MEASURED ketamine IC50 at rat GluN1/GluN2A in HEK293T cells at
#: -66 mV in 1 mM Mg2+, 5.35 +/- 0.34 uM [Kotermanski2009]. See "Which
#: ketamine affinity, and why it is not the tightest one" for why the
#: Mg2+-containing value is the one instantiated.
NMDA_KD_KETAMINE = 5350.0

#: nM. The same measurement in ZERO Mg2+, 0.33 +/- 0.01 uM
#: [Kotermanski2009], and [Dravid2007]'s Mg2+-free oocyte value at -40 mV,
#: 3.31 uM. Recorded, not instantiated: the 10-fold gap between two
#: Mg2+-free measurements 26 mV apart is this module's evidence that
#: ketamine's potency is voltage dependent in a way the model does not track.
NMDA_KD_KETAMINE_MG_FREE = 330.0
NMDA_KD_KETAMINE_MG_FREE_DRAVID = 3310.0

#: nM. MEASURED (+)MK-801 IC50 at rat NR1/NR2A, 0.015 uM, [Dravid2007]
#: Table 2, pH 7.6, -40 mV, Mg2+-free. Recorded and used only as the
#: consistency check on the rate pair below, which is what the site is
#: actually built from.
NMDA_IC50_MK801 = 15.0

#: MEASURED (+)MK-801 rate constants at rat NR1/NR2A from macroscopic
#: response relaxation, pH 7.6 [Dravid2007] Table 5, in the paper's own
#: units. Derivation 1: this module's only ligand with BOTH constants
#: measured.
DRAVID2007_MK801_KON_PER_M_PER_S = 3.9e6
DRAVID2007_MK801_KOFF_PER_S = 0.12

#: nM. [Dravid2007]'s own printed Kd for the pair above, and [Wong1986]'s
#: independent equilibrium binding Kd of [3H]MK-801 in rat brain membranes.
#: They agree to 13% across two laboratories, two preparations and two
#: methods; a test asserts that agreement and that the rate pair reproduces
#: the printed Kd.
DRAVID2007_MK801_KD = 33.0
WONG1986_MK801_BINDING_KD = 37.2

#: min. [Huettner1988]'s recovery time constants from MK-801 block during
#: continuous agonist: ~90 min at -70 to -80 mV and ~2 min at +30 mV. The
#: falsifier for this module's `koff`, not an input to it. See "The MK-801
#: unblocking rate this model gets wrong by 650-fold".
HUETTNER1988_MK801_RECOVERY_TAU_HYPERPOLARIZED = 90.0
HUETTNER1988_MK801_RECOVERY_TAU_DEPOLARIZED = 2.0

#: nM. MEASURED NBQX apparent Ki versus AMPA (63 nM) and versus kainate
#: (78 nM) in Xenopus oocytes expressing rat cortex mRNA [Randle1992]. The
#: near-equipotency is the reason "AMPA antagonist" here means "AMPA and
#: kainate antagonist"; a test asserts the two are within 25% of each other.
AMPA_KD_NBQX = 63.0
KAINATE_KD_NBQX = 78.0

#: [Randle1992]'s Schild pA2 versus AMPA (7.05 +/- 0.10) and versus kainate
#: (7.17 +/- 0.05), with slopes near unity. The pA2 is a second, independent
#: reading of the same affinity -- 10^(9 - 7.05) = 89 nM against the 63 nM
#: apparent Ki, a 1.4-fold spread -- and it is what establishes that NBQX's
#: antagonism is COMPETITIVE, which is why NBQX gets an orthosteric ligand
#: entry rather than a blocker pool.
RANDLE1992_NBQX_PA2_VS_AMPA = 7.05
RANDLE1992_NBQX_PA2_VS_KAINATE = 7.17

#: nM. MEASURED NBQX IC50 against kainate/AMPA-mediated field EPSPs in rat
#: hippocampal CA1 slices, 0.90 uM [Randle1992]: 14-fold weaker than the
#: same paper's oocyte Ki, an offset that paper reports is shared by other
#: quinoxalinediones. Recorded as the tissue-versus-expression-system error
#: bar on every affinity in this module, none of which was measured in
#: intact tissue.
AMPA_IC50_NBQX_HIPPOCAMPAL_SLICE = 900.0

#: nM. MEASURED subtype spreads, recorded and not instantiated (see "What was
#: deliberately not built"). [Dravid2007] Table 2, pH 7.6, and
#: [Kotermanski2009] in 0 and 1 mM Mg2+.
NMDA_SUBTYPE_IC50_MK801 = {"2A": 15.0, "2B": 9.0, "2C": 24.0, "2D": 38.0}
NMDA_SUBTYPE_IC50_KETAMINE_DRAVID = {
    "2A": 3310.0, "2B": 927.0, "2C": 1650.0, "2D": 2420.0
}
NMDA_SUBTYPE_IC50_KETAMINE_MG_FREE = {
    "2A": 330.0, "2B": 510.0, "2C": 1180.0, "2D": 2950.0
}
NMDA_SUBTYPE_IC50_KETAMINE_1MM_MG = {
    "2A": 5350.0, "2B": 5080.0, "2C": 1180.0, "2D": 2950.0
}


# ---------------------------------------------------------------------------
# Association rate constants
# ---------------------------------------------------------------------------

#: nM^-1 min^-1. MEASURED. [Dravid2007]'s (+)MK-801 k_ON, converted. Shared
#: by both channel blockers in the NMDA pool (derivation 3 applied within a
#: pool rather than within a site): ketamine and MK-801 are the same
#: mechanism at the same site, and ketamine's own k_ON was not measured
#: anywhere reached.
NMDA_CHANNEL_KON = kon_per_molar_second_to_per_nM_min(DRAVID2007_MK801_KON_PER_M_PER_S)

#: min^-1. MEASURED. [Dravid2007]'s (+)MK-801 k_OFF, converted. The one
#: dissociation rate constant in this module that is not derived from an
#: affinity -- and, per "The MK-801 unblocking rate this model gets wrong by
#: 650-fold", the one most likely to be wrong at physiological voltages.
NMDA_MK801_KOFF = per_second_to_per_min(DRAVID2007_MK801_KOFF_PER_S)

#: nM^-1 min^-1. ILLUSTRATIVE, and deliberately identical to
#: `acetylcholine.ILLUSTRATIVE_KON` (a test asserts the equality) for the
#: same two stated reasons: it sits at the slow end of the only measured kon
#: panel this codebase has ([Sykes2022] at beta1, P5.3), so it does not
#: assert unusually fast binding; and it is fast enough that binding
#: equilibrates in well under a minute at every ORTHOSTERIC affinity here.
#: Used for the two orthosteric pools only -- the NMDA channel-blocker pool
#: has a measured kon and does not need it.
ILLUSTRATIVE_KON = 0.01


# ---------------------------------------------------------------------------
# Efficacies
# ---------------------------------------------------------------------------

#: Glutamate's efficacy, 1.0, at both sites: it is the endogenous
#: transmitter and the reference full agonist every panel cited above
#: normalizes to. Note that in this module efficacy does NOT set how much
#: drive the receptor delivers -- the connectivity weight does that -- so
#: this constant enters only through which ligands count as antagonists.
GLUTAMATE_EFFICACY = 1.0

#: NBQX's efficacy, 0.0: a neutral competitive antagonist. Unlike ketanserin
#: (P5.1), SCH-23390 (P5.2), prazosin (P5.3) and scopolamine (P5.4), this is
#: NOT a simplification of an inverse agonist. An ionotropic glutamate
#: receptor is a ligand-gated channel with no constitutive activity for an
#: inverse agonist to reduce, so 0.0 is what a competitive antagonist at
#: this receptor is.
NBQX_EFFICACY = 0.0

#: Ketamine's and MK-801's efficacies are not used at all: they are channel
#: blockers, they never enter the orthosteric weighted sum, and these zeros
#: exist only so the `Ligand` constructor has a value.
KETAMINE_EFFICACY = 0.0
MK801_EFFICACY = 0.0


# ---------------------------------------------------------------------------
# The site: a receptor pool whose output is a fraction of an EXISTING weight
# ---------------------------------------------------------------------------


class GlutamatergicTransmissionSite(ChannelBlockingReceptorSite):
    """A glutamate receptor pool that REMOVES transmission rather than adding it.

    This class is the double-counting resolution in code. Its `activation` is
    not a drive, and it is not positive: it is

        activation = -synaptic_share * blocked_transmission_fraction

    i.e. the NEGATIVE of the fraction of an existing excitatory connectivity
    weight that the drugs on board remove, so that the engine's
    `connection_modulation` factor `1 + gain * activation` with
    `gain == GLUTAMATE_CONNECTION_GAIN == 1.0` is exactly
    `1 - (fraction removed)`. The gain is therefore not a free parameter and
    not an illustrative one; it is pinned by the algebra. See the module
    docstring.

    `synaptic_share` is this receptor's share of the weight -- `nmda_share`
    or `ampa_share`, which sum to 1 -- and is a constant of the site rather
    than a state, because it depends on the postsynaptic membrane potential
    and this model has no membrane potential.

    `blocked_transmission_fraction` composes the two mechanisms
    multiplicatively, so a receptor blocked by both routes is not counted
    twice::

        blocked = 1 - (1 - competitive) * (1 - channel_blocked)

    where `competitive` is the summed occupancy of the ORTHOSTERIC ligands
    whose efficacy is zero (NBQX at AMPA) and `channel_blocked` is the
    channel-blocker pool's occupancy (ketamine and MK-801 at NMDA). Agonist
    occupancy is NOT counted as block, which is the point of the distinction:
    raising glutamate displaces NBQX and lowers the block, and does nothing
    at all to ketamine. A test asserts both halves of that sentence, because
    the competitive/uncompetitive contrast is the reason the two mechanisms
    are represented differently at all.
    """

    def __init__(self, name, ligands, synaptic_share, channel_blockers=None):
        ChannelBlockingReceptorSite.__init__(
            self, name=name, ligands=ligands, channel_blockers=channel_blockers
        )
        if not 0.0 <= synaptic_share <= 1.0:
            raise ValueError(
                "synaptic_share is a fraction of one connectivity weight and "
                "must lie in [0, 1], got %r" % (synaptic_share,)
            )
        self.synaptic_share = float(synaptic_share)
        self._antagonist_indices = [
            i for i, lig in enumerate(self.orthosteric_ligands) if lig.efficacy == 0.0
        ]

    def competitive_block_fraction(self, state):
        """Summed occupancy of the orthosteric ligands with zero efficacy."""
        bound, _ = self.split_states(state)
        return float(sum(bound[i] for i in self._antagonist_indices))

    def blocked_transmission_fraction(self, state):
        """Fraction of this receptor's transmission removed by drug on board."""
        competitive = self.competitive_block_fraction(state)
        channel = self.blocked_fraction(state)
        return 1.0 - (1.0 - competitive) * (1.0 - channel)

    def removed_weight_fraction(self, state):
        """Fraction of the whole connectivity weight this site removes."""
        return self.synaptic_share * self.blocked_transmission_fraction(state)

    def activation(self, state):
        return -self.removed_weight_fraction(state)


#: The `connection_modulation` gain every glutamate system uses. NOT a free
#: parameter: `1 + gain * activation` has to equal `1 - fraction removed`,
#: and `activation` is already the negative of the fraction removed, so the
#: gain is 1.0 by construction. It exists as a named constant so that the
#: one place it could silently become a tuning knob is visible.
GLUTAMATE_CONNECTION_GAIN = 1.0


def _ligand(name, kd, efficacy, kon=ILLUSTRATIVE_KON):
    """Derivation 3: `koff = kon * Kd`, so the measured affinity is exact and
    only the approach timescale carries the illustrative `kon`."""
    return Ligand(name=name, kon=kon, koff=kon * kd, efficacy=efficacy)


def ligand_mk801():
    """(+)MK-801 with BOTH rate constants measured ([Dravid2007] Table 5).

    Derivation 1, and the only instance of it in this module. `koff/kon` is
    30.8 nM, i.e. it reproduces that table's own printed Kd of 33 nM to
    within 7% without being told to, and [Wong1986]'s independent binding
    Kd of 37.2 nM to within 17%.
    """
    return Ligand(
        name="mk801",
        kon=NMDA_CHANNEL_KON,
        koff=NMDA_MK801_KOFF,
        efficacy=MK801_EFFICACY,
    )


def ligand_ketamine():
    """Ketamine sharing MK-801's measured `kon` (derivation 3 within the
    channel-blocker pool), with `koff` set from its own measured IC50 in
    1 mM Mg2+."""
    return Ligand(
        name="ketamine",
        kon=NMDA_CHANNEL_KON,
        koff=NMDA_CHANNEL_KON * NMDA_KD_KETAMINE,
        efficacy=KETAMINE_EFFICACY,
    )


def ligands_nmda():
    """(orthosteric, blockers) for the NMDA site.

    Orthosteric: glutamate alone. There is no competitive NMDA antagonist in
    this module, and glutamate is carried NOT because its occupancy drives
    anything -- the connectivity weight does that -- but because the claim
    that ketamine and MK-801 are UNCOMPETITIVE is only testable against a
    pool the agonist actually occupies. Blockers: ketamine, MK-801.
    """
    orthosteric = [_ligand("glutamate", NMDA_KD_GLUTAMATE, GLUTAMATE_EFFICACY)]
    blockers = [ligand_ketamine(), ligand_mk801()]
    return orthosteric, blockers


def ligands_ampa():
    """glutamate / NBQX at the AMPA site, in that order, one competitive pool.

    NBQX genuinely competes for the agonist site -- [Randle1992]'s Schild
    slopes near unity are the measurement, not an assumption -- so this needs
    nothing `ReceptorSite` did not already do, exactly as P5.4 found for
    scopolamine at the muscarinic sites.
    """
    return [
        _ligand("glutamate", AMPA_KD_GLUTAMATE, GLUTAMATE_EFFICACY),
        _ligand("nbqx", AMPA_KD_NBQX, NBQX_EFFICACY),
    ]


def receptor_site_nmda(
    v_mV=ILLUSTRATIVE_OPERATING_VOLTAGE,
    ratio=NMDA_AMPA_CONDUCTANCE_RATIO,
    condition=DEFAULT_IONIC_CONDITION,
):
    """The NMDA pool on a glutamatergic connection, at operating potential
    `v_mV`. Its share of the weight is `nmda_share(v_mV)` and therefore
    collapses toward zero at resting potential."""
    orthosteric, blockers = ligands_nmda()
    return GlutamatergicTransmissionSite(
        name="NMDA_on_glutamatergic_connection",
        ligands=orthosteric,
        synaptic_share=nmda_share(v_mV, ratio=ratio, condition=condition),
        channel_blockers=blockers,
    )


def receptor_site_ampa(
    v_mV=ILLUSTRATIVE_OPERATING_VOLTAGE,
    ratio=NMDA_AMPA_CONDUCTANCE_RATIO,
    condition=DEFAULT_IONIC_CONDITION,
):
    """The AMPA pool on the same connection, carrying the rest of the weight."""
    return GlutamatergicTransmissionSite(
        name="AMPA_on_glutamatergic_connection",
        ligands=ligands_ampa(),
        synaptic_share=ampa_share(v_mV, ratio=ratio, condition=condition),
    )


def plain_ampa_binding_site():
    """The AMPA orthosteric pool as an ordinary `ReceptorSite`.

    Not used by any scenario. It exists so the competitive glutamate/NBQX
    equilibrium can be checked against the class every other receptor in this
    codebase uses, which is the cheapest available guard against
    `GlutamatergicTransmissionSite` having quietly changed the binding
    arithmetic while changing what is done with it.
    """
    return ReceptorSite(name="AMPA_orthosteric_pool", ligands=ligands_ampa())


# ---------------------------------------------------------------------------
# Pharmacokinetics
# ---------------------------------------------------------------------------

#: min^-1. ILLUSTRATIVE. ln(2) / 30 min, shared by ketamine, MK-801 and
#: NBQX, and deliberately the same number as `serotonin.TOOL_COMPOUND_KE`,
#: `dopamine.TOOL_COMPOUND_KE`, `noradrenaline.TOOL_COMPOUND_KE` and
#: `acetylcholine.TOOL_COMPOUND_KE`; a test asserts the equality. Do not cite
#: it as a half-life for any of the three.
TOOL_COMPOUND_KE = 0.023105

#: min^-1. ILLUSTRATIVE PROXY, the same value and the same reason as the four
#: transmitter clearance constants before it. Extracellular glutamate is
#: cleared by EAAT transporters on a millisecond timescale in the cleft, not
#: by a first-order process on a scale of seconds; 5.0 min^-1 (t1/2 ~8.3 s)
#: preserves the only property the dynamics need -- that the transmitter
#: clears far faster than any dosed drug -- at a timescale the integrator
#: resolves. The real quantity belongs to P6.2.
GLUTAMATE_KE = 5.0


def pk_models_nmda():
    """One PK model per NMDA ligand, in site ligand order
    (glutamate, ketamine, mk801)."""
    return [
        OneCompartmentPK(ke=GLUTAMATE_KE),
        OneCompartmentPK(ke=TOOL_COMPOUND_KE),
        OneCompartmentPK(ke=TOOL_COMPOUND_KE),
    ]


def pk_models_ampa():
    """One PK model per AMPA ligand (glutamate, nbqx)."""
    return [
        OneCompartmentPK(ke=GLUTAMATE_KE),
        OneCompartmentPK(ke=TOOL_COMPOUND_KE),
    ]


# ---------------------------------------------------------------------------
# The circuit: the same cortical loop P5.1, P5.3 and P5.4 use
# ---------------------------------------------------------------------------
#
# Two nodes, and NO new illustrative weight:
#
#   PFC_Pyramidal   layer V/VI prefrontal pyramidal neurons.
#   PFC_GABA        local cortical GABAergic interneurons.
#
# The weights, gain and baseline inputs are character-for-character those of
# `serotonin.raphe_cortical_network`, `noradrenaline.
# locus_coeruleus_cortical_network` and `acetylcholine.
# basal_forebrain_cortical_network`, minus each of those modules' isolated
# source-nucleus node, which this module has no use for. Reusing them is the
# point: this module's result has to be a consequence of the connectivity
# every other P5 scenario already ran on, not of a loop drawn to produce it.
#
# The three rules, unchanged from P5.3:
#
#   1. Baselines put every node at rate 0.5 at rest with no drug,
#      `stability.MAX_SENSITIVITY_RATE`. With the loop weights below that
#      fixes baseline_input at +0.45 for the pyramidal node and -0.4 for the
#      interneuron exactly, and `stability.balanced_baselines` reproduces
#      them.
#   2. contraction_number = 0.9 < 1, so the network has exactly one fixed
#      point (P3.4's global-uniqueness certificate).
#   3. No node saturates. This module cannot violate rule 3 the way the
#      others could, because it adds no drive to any node at all -- the
#      strongest possible form of the check, and a test states it that way.
#
# EXACTLY ONE of the two connections is glutamatergic: PFC_Pyramidal ->
# PFC_GABA, weight +0.8. The return connection, PFC_GABA -> PFC_Pyramidal at
# -0.9, is GABAergic and this module does not touch it. That asymmetry is
# not a choice made here; it is what the signs of the connectivity matrix
# already mean.

#: ILLUSTRATIVE. Unchanged from every hand-built cortical loop in this
#: codebase.
CORTICAL_INHIBITION_WEIGHT = -0.9
CORTICAL_RECURRENT_WEIGHT = 0.8

#: ILLUSTRATIVE, and the same value every hand-built scenario uses.
CORTICAL_GAIN = 4.0

#: The node parameters. `baseline_input` is fully determined by rule 1.
_NODES = (
    # name,           tau,  baseline_input
    ("PFC_Pyramidal", 1.5, 0.45),
    ("PFC_GABA", 1.0, -0.4),
)

#: The one glutamatergic connection in the loop, as the engine's
#: `connection_modulation` key: (postsynaptic, presynaptic).
GLUTAMATERGIC_CONNECTION = ("PFC_GABA", "PFC_Pyramidal")

#: ILLUSTRATIVE. The recurrent pyramidal -> pyramidal weight used ONLY by
#: `cortical_network(recurrent_excitation=...)`, which exists to answer the
#: question the two-node loop cannot: whether the disinhibition direction
#: survives when NMDA block also removes recurrent excitation. Zero by
#: default, so the shared network is untouched.
RECURRENT_EXCITATION_CONNECTION = ("PFC_Pyramidal", "PFC_Pyramidal")


def cortical_network(recurrent_excitation=0.0):
    """PFC_Pyramidal -| PFC_GABA cortical loop.

    `recurrent_excitation` adds a pyramidal -> pyramidal self-weight, which
    is zero by default. It is not part of any scenario; it exists so a test
    can build the variant in which NMDA block removes recurrent excitation
    as well as feedforward drive onto the interneuron, and measure where the
    net direction on the pyramidal node flips. Note that a non-zero value
    moves the baseline fixed point, so `rest_rates` no longer applies to it
    and `stability.balanced_baselines` must be re-run; the test does that.
    """
    nodes = [
        CircuitNode(name=name, tau=tau, baseline_input=baseline, gain=CORTICAL_GAIN)
        for name, tau, baseline in _NODES
    ]
    idx = {node.name: i for i, node in enumerate(nodes)}
    n = len(nodes)
    W = np.zeros((n, n))
    W[idx["PFC_Pyramidal"]][idx["PFC_GABA"]] = CORTICAL_INHIBITION_WEIGHT
    W[idx["PFC_GABA"]][idx["PFC_Pyramidal"]] = CORTICAL_RECURRENT_WEIGHT
    if recurrent_excitation:
        W[idx["PFC_Pyramidal"]][idx["PFC_Pyramidal"]] = float(recurrent_excitation)
    return CircuitNetwork(nodes=nodes, connectivity=W)


def rest_rates(network=None):
    """The rate vector rule 1 puts every node at with no drug on board.

    Starting there rather than at zero matters more in this module than in
    any before it: with no drug the receptor layer's `activation` is exactly
    zero and the network is EXACTLY the drug-free one, so any deviation from
    0.5 in a control run is integrator noise and can be asserted as such.
    """
    n = len(_NODES) if network is None else network.n
    return np.full(n, 0.5)


# ---------------------------------------------------------------------------
# Receptor systems: no coupling, only connection modulation
# ---------------------------------------------------------------------------


def _connection_modulation(connections=None):
    """`{(post, pre): 1.0}` for each glutamatergic connection modulated."""
    if connections is None:
        connections = [GLUTAMATERGIC_CONNECTION]
    return {key: GLUTAMATE_CONNECTION_GAIN for key in connections}


def receptor_system_nmda(
    v_mV=ILLUSTRATIVE_OPERATING_VOLTAGE,
    ratio=NMDA_AMPA_CONDUCTANCE_RATIO,
    condition=DEFAULT_IONIC_CONDITION,
    connections=None,
):
    """The NMDA system. `coupling` is EMPTY, on purpose: see the module
    docstring's double-counting section."""
    return ReceptorSystem(
        pk_models=pk_models_nmda(),
        receptor_site=receptor_site_nmda(v_mV=v_mV, ratio=ratio, condition=condition),
        coupling={},
        connection_modulation=_connection_modulation(connections),
        name="NMDA",
    )


def receptor_system_ampa(
    v_mV=ILLUSTRATIVE_OPERATING_VOLTAGE,
    ratio=NMDA_AMPA_CONDUCTANCE_RATIO,
    condition=DEFAULT_IONIC_CONDITION,
    connections=None,
):
    """The AMPA system, same connection, same empty coupling."""
    return ReceptorSystem(
        pk_models=pk_models_ampa(),
        receptor_site=receptor_site_ampa(v_mV=v_mV, ratio=ratio, condition=condition),
        coupling={},
        connection_modulation=_connection_modulation(connections),
        name="AMPA",
    )


def glutamatergic_receptor_systems(
    v_mV=ILLUSTRATIVE_OPERATING_VOLTAGE,
    ratio=NMDA_AMPA_CONDUCTANCE_RATIO,
    condition=DEFAULT_IONIC_CONDITION,
    connections=None,
):
    """Both systems, NMDA then AMPA, for `Simulation`.

    They modulate the SAME connection, and the engine composes their factors
    multiplicatively: `W_eff = W * (1 - s_N*b_N) * (1 - s_A*b_A)`. The
    additive partition in the module docstring is the first-order form of
    that, and the two differ only by the second-order cross term
    `s_N*b_N*s_A*b_A`, which is nonzero only when an NMDA blocker and an AMPA
    blocker are on board at once. A test measures that cross term rather
    than assuming it away, and it is the one place where the composition rule
    documented in `engine.Simulation` is load-bearing here.

    Glutamate appears once per system, as two separate `Ligand` objects with
    two separate concentration states, exactly as acetylcholine does across
    P5.4's four systems. `Simulation.dose_vector(glutamate=...)` matches by
    name across both, which is why both systems must give glutamate the SAME
    `ke`; a test asserts that.
    """
    return [
        receptor_system_nmda(
            v_mV=v_mV, ratio=ratio, condition=condition, connections=connections
        ),
        receptor_system_ampa(
            v_mV=v_mV, ratio=ratio, condition=condition, connections=connections
        ),
    ]


# ---------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------


def cortical_output_readout():
    """Layer V/VI prefrontal pyramidal firing.

    Deliberately the same name, node and weight as
    `serotonin.cortical_output_readout`, `noradrenaline.
    cortical_output_readout` and `acetylcholine.cortical_output_readout`; a
    test asserts all four are equal. This is the "RS neurons" side of
    [Homayoun2007].
    """
    return BehaviorReadout(name="cortical_output", weights={"PFC_Pyramidal": 1.0})


def cortical_interneuron_readout():
    """Local cortical GABAergic interneuron firing: the "FS neurons" side of
    [Homayoun2007], and the node whose fall is the mechanism rather than the
    result."""
    return BehaviorReadout(name="cortical_interneuron", weights={"PFC_GABA": 1.0})
