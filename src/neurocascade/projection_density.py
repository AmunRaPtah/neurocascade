"""P6.3: projection densities derived from the Phase 3 connectome.

    from neurocascade import projection_density as pd

    table = pd.load_table()                 # the committed derived table
    print(table.report())
    rho = table.density("CP", "VTA")        # nucleus VTA into region CP

P6.1 built the projection-system abstraction and gave every
`ProjectionTarget` a density of 1.0, tagged `provenance="illustrative"`,
with the note that "P6.3 is the task that derives real densities from the
Phase 3 connectome". This module is that derivation, and it makes exactly
one number per (source nucleus, target region) pair out of data this
project already ingested and checksum-verified in P3.1.

The formula, and why it is `connection_density` and not one of the other
three published normalizations
------------------------------------------------------------------------

For source nucleus `s` (one or more CCF structures) and target region `k`
(one or more CCF structures)::

    rho(s -> k) = sum_{i in s} sum_{j in k} strength(i, j) / sum_{j in k} voxels(j)

which is Knox et al. 2019's `connection_density` computed at whatever
grouping the caller asks for, exactly as `connectome/build.py` computes it
for the circuit layer. `volumes.py` recovers the voxel counts from the
published matrices themselves, so no new source and no new number enters
the project here.

Both halves of that choice are forced by what a projection density is FOR.
`projection.ProjectionSystem` uses `rho_k` as the coefficient of a release
rate in nM/min, i.e. an amount of transmitter per unit time per unit of
TARGET TISSUE:

* **Divide by the target's size.** Concentration is amount per volume, and
  the amount released into region `k` per unit time is proportional to the
  number of release sites there, so what the mass balance needs is release
  sites per unit target volume. That is `s_ij / |Y_j|`.
* **Do NOT divide by the source's size.** The source term is
  `release_scale * rho * L(r)` where `L` is the DIMENSIONLESS
  Wilson-Cowan rate ratio, so the number of neurons doing the releasing has
  to be inside `rho`. Knox's `s_ij` is summed over the source's voxels and
  therefore already carries it; dividing by `|X_i|` would say a large
  nucleus and a small one deliver the same transmitter at the same firing
  rate. This is `connectome/build.py`'s decision 1 restated for a
  concentration rather than for a drive.

Only ratios are used. `ProjectionSystem` fixes the absolute scale by
requiring the REFERENCE region to reproduce its clearance object's measured
basal concentration, so multiplying this whole table by a constant changes
nothing (`test_projection_density.py` asserts that).

The reference region is now the region the measurement was made in
------------------------------------------------------------------

This is the second thing P6.3 changes, and it is not cosmetic. Under P6.1's
uniform map every region's predicted concentration equalled the reference's,
so which region was the reference did not matter and the default (the
densest target) was harmless. With a real density map it matters completely:
the reference is the one region whose concentration is a MEASUREMENT and
every other region's is a PREDICTION relative to it. So each transmitter's
reference region is the CCF region its cited basal concentration was
measured in, and `REFERENCE_REGION` records the pairing:

    dopamine       CP    [Bosse2011]    mouse caudate-putamen, zero net flux
    serotonin      CP    [Mathews2004]  mouse striatum, zero net flux
    noradrenaline  PFC   [PanLai1995]   rat medial prefrontal cortex, ZNF
    acetylcholine  CP    [Vinson1997]   rat striatum, no-net-flux
    histamine      MEZ   [Itoh1991]     rat medial hypothalamus, recovery-
                                        corrected dialysate

Two of those force a region into a port that had no node for it, which is
legitimate and is what `ProjectionTarget` was built to allow: "a projection
can maintain a transmitter pool in a region this scenario carries no node
for". The serotonin and acetylcholine ports gain a caudoputamen pool, and
the histamine port gains a medial-hypothalamus pool.

What is DERIVED and what is not: the nuclear block
--------------------------------------------------

A region that is itself a source nucleus of the system -- the raphe for
serotonin, the locus coeruleus for noradrenaline, VTA and SNc for dopamine,
the three cholinergic nuclei, the tuberomammillary nucleus -- is NOT a
terminal field, and its density is NOT derived here. Two independent
reasons, in the order they matter:

**1. Mechanism.** The extracellular transmitter at a monoamine nucleus comes
from SOMATODENDRITIC release: vesicular release from the dendrites of the
resident neurons. An anterograde tracer measures axonal arbour. Substituting
one for the other would be substituting a different mechanism's measurement,
which is the same category error `projection_systems.py` refuses when it
declines to use Cumming et al. 1991's histamine pool-run-down half-life as a
clearance rate (that citation lives there, next to the constant it is about).

**2. The model refuses the numbers.** Derived anyway, the tracer's
nucleus-block entries are 7 to 2000 times the same nucleus's density into
its measured reference region:

    nucleus   rho(self) / rho(this transmitter's reference region)
    VTA          9.8      SNc        7.2
    DR         931        CS       880
    LC        1751        NDB      102
    PPN         23        LDT     1543
    TM           1.27

At uniform uptake kinetics that is a predicted extracellular concentration
of 931 x 2.8 nM = 2.6 uM of serotonin in the dorsal raphe, and a release
rate of 20,200 nM/min against SERT's measured `Vmax` of 4,800 nM/min, so
`concentration_for_uptake` raises: there is NO finite steady state. A model
whose resting state does not exist is not a model of a resting brain, so
this is the density map being falsified rather than an inconvenience.
`test_projection_density.py` asserts the refusal on the real numbers.

The tuberomammillary row is the exception that shows this argument is about
the DIAGONAL and not about nuclei: histamine's reference region is the
medial hypothalamus immediately around the nucleus, so the ratio is 1.27 and
nothing would have gone wrong numerically. TM is excluded anyway, because
reason 1 applies to it exactly as it applies to the other eight and a rule
with a numerical exception in it is not the same rule.

So each nuclear region carries ONE ILLUSTRATIVE density, chosen to be the
reference region's total, i.e. **the source nucleus's own extracellular
concentration is set equal to the reference region's measured one**. That
is not a new number: it is exactly what P6.1's uniform map produced for
those regions, kept and relabelled `illustrative-somatodendritic` so that
the one undelivered density in this task is visible in
`ProjectionSystem.density_report()` rather than buried. What it costs is
swept in the tests.

**Four measurements bear on it, three of them agree with it to within a
factor of two, and none of them is a calibration of it.** The rule sets
every nucleus:terminal-field concentration ratio to exactly 1. What the
literature says about that number:

    transmitter    ratio  design                                  source
    serotonin      1.86   DRN vs NAcc, ZERO NET FLUX, one study   [Tao2000]
    serotonin      1.88   DRN vs frontal ctx, dialysate, one      [Portas1998]
                          study, the independent replication
    serotonin      ~1     six regions, "no significant            [Adell1991]
                          differences among the brain regions"
    dopamine       0.22   VTA/medial SN 0.79 nM vs NAcc 3.62 nM,  [KalivasDuffy1993]
                          both NO NET FLUX, companion papers,
                          one laboratory and one protocol
    dopamine       0.23   SNr vs striatum, one paper, each        [Robertson1991]
                          region's dialysate divided by its own
                          probe recovery (raw ratio 0.12)
    dopamine       1.5    VTA 5.0 nM by NO NET FLUX against       [Engleman2011]
                          NAcc 3.96 nM, same laboratory,
                          different rat line
    noradrenaline  ~1     LC and cingulate cortex of the SAME     [Mateo2001]
                          rats, both "in the range 3-6 nM"

So the assumption is not "unknown". Every measurement lies between 0.22 and
1.9, i.e. within a factor of about 5 of the value the rule uses, in a
quantity whose tracer-derived alternative would have been 7 to 2000. It
does not become a calibration, for reasons that differ per row:
[Tao2000]'s terminal field is the nucleus accumbens and not the
caudoputamen this model's reference region is, and its full text could not
be reached to confirm which of its numbers are the zero-net-flux
intercepts; [Portas1998] reports fmol/sample rather than concentration and
its two probes' membrane lengths could not be checked, and [Sacchetti1999]
measured an 8-fold swing in a "basal" level from probe geometry alone;
[Adell1991] contradicts both; [Mateo2001] gives one range covering both
regions rather than two numbers, from probes of 2.0 mm and 4.0 mm.

And DOPAMINE's own three rows disagree with each other by 7-fold, which is
the reason no per-transmitter number is taken from them either.
[KalivasDuffy1993] and [Engleman2011] are both primary no-net-flux
measurements of basal dopamine in the rat ventral tegmental area and they
report 0.79 nM and 5.0 nM; the extraction fractions (0.60 and 0.46) are the
same order, so the gap is not a fitting artefact, and the two differ in rat
line, sex, perfusion rate and exactly which midbrain structures the probe
spanned. Averaging them would manufacture a number neither laboratory
measured. Rule 3 says the label comes off in the commit that supplies the
calibration for that parameter; this supplies support, not a number, so the
label stays and the support is written down.

**Acetylcholine is the row where the MECHANISM argument itself is doubtful,
and that is worse than a missing number.** [Consolo1990] measured
acetylcholine in the basal nuclear complex of freely-moving rats and
concluded that it "is largely axonal in nature" and "derives substantially
from axons of the cholinergic pontomesencephalic tegmentum". If that is
right, the pool at the basal forebrain is not somatodendritic release at
all but a terminal field of the PPT/LDT, i.e. exactly the quantity the
tracer DOES measure, and the nuclear block excludes a number it should have
derived. The exclusion is kept anyway, because keeping one rule for all
five nuclear regions is worth more than a per-transmitter exception built
on a single paper whose full text could not be reached, and because
[Consolo1990]'s own measurement was made with physostigmine in the
perfusate. It is recorded here as the sharpest known objection to this
module's second-largest decision.

What the derived map does NOT know: cell type
---------------------------------------------

The Knox et al. 2019 regionalized model is fitted to wild-type anterograde
injections, so a row of it is the projection of EVERY neuron in that
nucleus, not of its neuromodulatory subtype. None of the nine is a purely
modulatory nucleus, and whatever non-modulatory projections each has are in
its row. Using these rows as neuromodulatory projection densities assumes
the modulatory subtype's target distribution matches the nucleus's overall
one. That assumption is stated here rather than hidden, because it is the
leading suspect for the two validations that fail (see
`validation_report()`, its serotonin and acetylcholine rows, and the
per-nucleus diagnosis under the serotonin one).

Allen publishes Cre-line-specific tracer experiments that would remove the
assumption. They are not in this repository's manifest -- P3.1 ingested the
wild-type regionalized model -- and ingesting them is a data-layer task of
P3.1's size, not a paragraph of this one.

**One published map lets the assumption be MEASURED rather than argued
about, and it survives.** [Lin2023] mapped histaminergic fibres whole-brain
in HDC-CreERT2 mice at fMOST resolution and registered the result to Allen
CCFv3, i.e. it is cell-type specific, mouse, per-region, and in the same
ontology as the connectome. Over the 32 regions its own text names, the
derived tuberomammillary projection density and its measured fibre density
have a SPEARMAN RANK CORRELATION OF 0.83 (p = 3.5e-9; 0.83 again with every
hypothalamic region dropped, so the authors' injection-site exclusion is not
carrying it; Pearson on the logs, 0.85). `fibre_density_report()` computes
it. That is the single strongest statement this module can make: for the one
transmitter where a cell-type-specific whole-brain map exists to check it
against, deriving a projection density from a wild-type tracer row
reproduces the measured ordering.

It is ONE transmitter, and it does not license the assumption for the raphe
or for the cholinergic nuclei, whose rows carry a larger non-modulatory
fraction. The two-region ratio checks in `validation_report()` are where
those show up, and both of theirs fail.
"""

import csv
import os

import numpy as np

CITATIONS = {
    "Knox2019": (
        "Knox JE, Harris KD, Graddis N, Whitesell JD, Zeng H, Harris JA, "
        "Shea-Brown E, Mihalas S. High-resolution data-driven model of the "
        "mouse connectome. Netw Neurosci 2019;3(1):217-236. The regionalized "
        "connection_strength and connection_density matrices this table is "
        "derived from; the manifest entry, URL and sha256 are in "
        "connectome/sources.py."
    ),
    "Bosse2011": (
        "Bosse KE et al. J Neurochem 2011;120:385-395, PMC3385875. Mouse "
        "caudate-putamen basal extracellular dopamine by zero net flux, "
        "5.0 +/- 0.2 nM. The dopamine reference region is CP because this is "
        "where the number was measured."
    ),
    "Mathews2004": (
        "Mathews TA et al. J Neurosci Methods 2004;140:169-181, "
        "PMID:15589347. Mouse, zero net flux: basal extracellular serotonin "
        "2.8 +/- 1 nM in STRIATUM and 1.4 +/- 0.4 nM in FRONTAL CORTEX, in "
        "the same wild-type animals by the same method. The striatal value "
        "pins this model's serotonin reference region; the frontal-cortex "
        "value is then a held-out measurement the density map PREDICTS, "
        "which is why it is the sharpest validation in this module."
    ),
    "PanLai1995": (
        "Pan WH, Lai YJ. Synapse 1995;21:85-92, PMID:8525466. Conscious rat "
        "medial prefrontal cortex by zero net flux: noradrenaline "
        "1.59 +/- 0.37 nM. The noradrenaline reference region."
    ),
    "Vinson1997": (
        "Vinson PN, Justice JB Jr. J Neurosci Methods 1997;73:61-67, "
        "PMID:9130679. Rat striatum, no-net-flux quantitative microdialysis, "
        "no cholinesterase inhibitor: acetylcholine 18.4 +/- 11.8 nM. The "
        "acetylcholine reference region, with the caveat "
        "`projection_systems.ACETYLCHOLINE_BASAL` already carries: striatal "
        "acetylcholine comes largely from tonically active striatal "
        "INTERNEURONS rather than from the three projection nuclei this "
        "model's acetylcholine field is built from."
    ),
    "Itoh1991": (
        "Itoh Y, Oishi R, Nishibori M, Saeki K. J Neurochem 1991;56:769-774. "
        "Anaesthetized rat MEDIAL HYPOTHALAMUS: extracellular histamine "
        "'assumed to be 7.8 nM, by a calculation from in vitro recovery "
        "through the dialysis membrane'. The histamine reference region is "
        "the CCF's medial hypothalamic zone MINUS the tuberomammillary "
        "nucleus itself; see MEZ_TERMINAL_FIELD."
    ),
    "Devoto2005": (
        "Devoto P, Flore G, Saba P, Fa M, Gessa GL. Co-release of "
        "noradrenaline and dopamine in the cerebral cortex elicited by "
        "single train and repeated train stimulation of the locus coeruleus. "
        "BMC Neurosci 2005;6:31, PMC1134661. Male Sprague-Dawley RAT, "
        "anaesthetized, uncorrected dialysate, and the reason it is used "
        "here: the SAME 4 mm membrane, the same 2.2 uL/min flow and the same "
        "assay in all three regions, so its cross-region comparison is "
        "matched. 'Mean extracellular basal levels of NA and DA, expressed "
        "as pg/sample (40 ul) injected on column were: in the mPFC (n = 25) "
        "4.26 +/- 0.49 and 4.34 +/- 0.46, in the Occ (n = 26) 4.90 +/- 0.65 "
        "and 4.22 +/- 0.58, in the caudate nucleus (n = 6) 1.71 +/- 0.16 and "
        "27.35 +/- 3.35 for NA and DA, respectively.' Gives TWO of the four "
        "checks below: noradrenaline mPFC:caudate = 2.49 and dopamine "
        "caudate:mPFC = 6.30."
    ),
    "Ihalainen1999": (
        "Ihalainen JA, Riekkinen P Jr, Feenstra MG. Neurosci Lett "
        "1999;277:71-74. MOUSE microdialysate: 'Basal release of NA was "
        "similar in these areas, but DA in striatum was 13-30 times higher "
        "than in other areas' (prefrontal cortex, striatum, hippocampus). "
        "The independent check on the derived DOPAMINE map's "
        "striatum:cortex ratio."
    ),
    "Mateo2001": (
        "Mateo Y, Fernandez-Pastor B, Meana JJ. Acute and chronic effects of "
        "desipramine and clorgyline on alpha2-adrenoceptors regulating "
        "noradrenergic transmission in the rat brain: a dual-probe "
        "microdialysis study. Br J Pharmacol 2001;133:1362-1370, PMC1621148. "
        "Male Sprague-Dawley rat, freely moving, TWO probes in the same "
        "animal: 'One probe (exposed tip 2.0 mm x 0.25 mm) was implanted in "
        "the vicinity of the right LC' and 'the other one (exposed tip "
        "4.0 mm x 0.25 mm) in the ipsilateral cingulate cortex'. 'Basal "
        "extracellular NA concentrations measured in the LC and in the "
        "cingulate cortex of control groups were in the range 3-6 nM.' "
        "'Values were not corrected for in vitro recovery.' The only "
        "within-animal nucleus-versus-terminal-field comparison found for "
        "any of the five transmitters."
    ),
    "Tao2000": (
        "Tao R, Ma Z, Auerbach SB. Differential effect of local infusion of "
        "serotonin reuptake inhibitors in the raphe versus forebrain. "
        "J Pharmacol Exp Ther 2000;294:571-579, PMID:10900234. Rat, "
        "unanaesthetized, and the only WITHIN-STUDY zero-net-flux "
        "nucleus:terminal-field comparison found for any transmitter: 'The "
        "larger increase in 5-HT in raphe was confirmed using zero-net-flux "
        "microdialysis. In NAcc, baseline 5-HT was 0.7 nM ... Baseline 5-HT "
        "in DRN was greater, 1.3 nM'. Ratio 1.86. Full text closed, so which "
        "of the two numbers are the zero-net-flux intercepts could not be "
        "confirmed from the Methods; the terminal field is the nucleus "
        "accumbens rather than the caudoputamen."
    ),
    "Portas1998": (
        "Portas CM, Bjorvatn B, Fagerland S, Gronli J, Mundal V, Sorensen E, "
        "Ursin R. Neuroscience 1998;83:807-814, PMID:9483564. Freely-moving "
        "rat, dorsal raphe and frontal cortex: waking means 3.2 +/- 0.9 and "
        "1.7 +/- 0.4 fmol/sample, ratio 1.88, i.e. an independent "
        "replication of [Tao2000]'s ratio by a different method. "
        "fmol/sample rather than concentration, and the two probes' membrane "
        "lengths could not be checked."
    ),
    "Adell1991": (
        "Adell A, Carceller A, Artigas F. Regional distribution of "
        "extracellular 5-hydroxytryptamine and 5-hydroxyindoleacetic acid in "
        "the brain of freely moving rats. J Neurochem 1991;56:709-712, "
        "PMID:1703223. Six regions including the raphe: 'The extracellular "
        "levels of 5-HT showed no significant differences among the brain "
        "regions studied.' The source that CONTRADICTS [Tao2000] and "
        "[Portas1998], recorded rather than dropped."
    ),
    "KalivasDuffy1993": (
        "Kalivas PW, Duffy P. Time course of extracellular dopamine and "
        "behavioral sensitization to cocaine. I. Dopamine axon terminals "
        "(J Neurosci 1993;13:266-275, PMC6576317) and II. Dopamine perikarya "
        "(1993;13:276-284, PMC6576314). Freely-moving male Sprague-Dawley "
        "rats, QUANTITATIVE NO-NET-FLUX microdialysis in both papers, one "
        "laboratory and one protocol: daily-saline controls give "
        "0.79 +/- 0.14 nM in the VTA/medial substantia nigra and "
        "3.62 +/- 0.50 nM in the nucleus accumbens on day 7, i.e. a "
        "nucleus:terminal ratio of 0.22. Extraction fractions 0.60 and 0.26. "
        "Companion papers rather than one paper, so different animals."
    ),
    "Robertson1991": (
        "Robertson GS, Damsma G, Fibiger HC. J Neurosci 1991;11:2209-2216, "
        "PMC6575457. Freely-moving Wistar rat, substantia nigra pars "
        "reticulata AND striatum in the SAME study: basal dopamine "
        "0.61 +/- 0.12 against 5.13 +/- 0.58 fmol/min, with the authors' own "
        "caveat that 'the exposed length of the nigral probe was "
        "approximately 50% that of the striatal probe' and in vitro "
        "recoveries of 6.8% and 13.1%. Dividing each region by its own "
        "recovery gives 0.23."
    ),
    "Engleman2011": (
        "Engleman EA, Keen EJ, Tilford SS, Thielen RJ, Morzorati SL. Alcohol "
        "2011;45:549-557, PMC3154620. Freely-moving alcohol-naive P rats, "
        "NO-NET-FLUX quantitative microdialysis in the posterior ventral "
        "tegmental area: 'The basal extracellular DA concentrations and DA "
        "extraction fractions in the posterior VTA were, respectively, "
        "(mean +/- SEM) 5.0 +/- 0.5 nM and 46 +/- 4%'. Selectively bred "
        "line, female only, and no terminal field in the same study."
    ),
    "Franklin2009": (
        "Franklin KM et al. Alcohol Clin Exp Res 2009;33:1721, PMC2858589. "
        "Female Wistar nucleus accumbens by NO NET FLUX in the same "
        "laboratory as [Engleman2011]: 'basal DA levels in the NAc "
        "(3.96 +/- 0.42 nM), compared with EtOH-naive animals "
        "(3.25 +/- 0.23 nM)'. The terminal-field anchor [Engleman2011] "
        "lacks; different rat line and probe length, so the 1.5 it implies "
        "is a cross-study ratio."
    ),
    "Consolo1990": (
        "Consolo S, Bertorelli R, Forloni GL, Butcher LL. Neuroscience "
        "1990;37:717-723, PMID:2247220. Acetylcholine measured IN the basal "
        "nuclear complex of freely-moving rats, concluding that it 'is "
        "largely axonal in nature' and 'derives substantially from axons of "
        "the cholinergic pontomesencephalic tegmentum'. The sharpest known "
        "objection to treating the basal forebrain's acetylcholine pool as "
        "somatodendritic. Measured with physostigmine in the perfusate."
    ),
    "Pudovkina2001": (
        "Pudovkina OL, Kawahara Y, de Vries J, Westerink BH. Brain Res "
        "2001;906:38-45, PMID:11430860. Dual-probe locus coeruleus and "
        "prefrontal cortex in conscious rats, and one of the four locus "
        "coeruleus studies checked for an absolute basal concentration: it "
        "reports percentages of control only. Its sister paper (Kawahara Y, "
        "Kawahara H, Westerink BHC, Naunyn-Schmiedebergs Arch Pharmacol "
        "1999;360:42-49, PMID:10463333) states that 'samples from the LC "
        "were collected in the presence of nomifensine ... 50 microM', so "
        "this laboratory's locus coeruleus values are reuptake-blocked at "
        "one probe only and are not comparable across regions even if the "
        "numbers were reachable."
    ),
    "VanGaalen1997": (
        "Van Gaalen M, Kawahara H, Kawahara Y, Westerink BHC. Brain Res "
        "1997;763:56-62, PMID:9272828. Dual probe locus coeruleus to "
        "prefrontal cortex; the locus coeruleus probe delivers drug and only "
        "the cortex is sampled, and the data are percentages of control. "
        "Checked and empty for an absolute value; useful for one qualitative "
        "anchor, that tetrodotoxin or clonidine into the locus coeruleus "
        "drops cortical noradrenaline to about 20%."
    ),
    "FernandezPastor2005": (
        "Fernandez-Pastor B, Mateo Y, Gomez-Urquijo S, Meana JJ. "
        "Psychopharmacology 2005;180:570-579, PMID:15717207. Awake rats, "
        "locus coeruleus and cingulate cortex, i.e. a direct replication of "
        "[Mateo2001]'s design. The abstract reports percentage changes only "
        "and the full text was not reachable. Checked and empty."
    ),
    "Walling2004": (
        "Walling SG, Nutt DJ, Lalies MD, Harley CW. J Neurosci "
        "2004;24:7421-7426, PMC6729640. Urethane-anaesthetized rat; the "
        "locus coeruleus is infused and the dentate gyrus is sampled, so it "
        "carries no locus coeruleus concentration. Checked and empty."
    ),
    "Sacchetti1999": (
        "Sacchetti G, Bernini M, Bianchetti A, Parini S, Invernizzi RW, "
        "Samanin R. Br J Pharmacol 1999;128:1332-1338, PMC1571760. "
        "'Hippocampal basal levels of NA (fmol/40 ul) are approximately 23 "
        "and 3 respectively using the horizontal and vertical probes', i.e. "
        "an 8-fold swing in a basal level from probe geometry alone, within "
        "one region. The reason an uncorrected dialysate ratio between two "
        "regions with different probes is not a concentration ratio."
    ),
    "Abercrombie1988": (
        "Abercrombie ED, Keller RW Jr, Zigmond MJ. Neuroscience "
        "1988;27:897-904, PMID:3252176. Freely-moving rat hippocampus, "
        "in-vitro-recovery-corrected (21%): approximately 14 nM. Already "
        "cited by `projection_systems.py` as the 8.8-fold cross-check on "
        "[PanLai1995]; used here only to size the cross-laboratory spread."
    ),
    "Lin2023": (
        "Lin W, Xu L, Zheng Y, An S, Zhao M, Hu W, Li M, Dong H, Li A, Li Y, "
        "Gong H, Pan G, Wang Y, Luo Q, Chen Z. Whole-brain mapping of "
        "histaminergic projections in mouse brain. PNAS "
        "2023;120(14):e2216231120, PMC10083611, open access. MOUSE "
        "HDC-CreERT2 (so the labelled fibres ARE the histaminergic ones), "
        "AAV-CAG-FLEX-ArchT-GFP into the TMN, n = 3, fMOST at "
        "0.32 x 0.32 x 2 um, REGISTERED TO ALLEN CCFv3. The metric is the "
        "percentage of a CCF region's pixels that are labelled. Grading: "
        "'Dense (value >= 0.5%), Moderate (0.1% <= value < 0.5%), and Few "
        "(value < 0.1%)', giving '35 brain regions were dense, 204 brain "
        "regions were moderate, and 326 brain regions were few'. SI Table "
        "S2 lists the 24 regions containing the injection site, which the "
        "authors exclude from their grading and which are excluded here."
    ),
    "Mechawar2000": (
        "Mechawar N, Cozzari C, Descarries L. Cholinergic innervation in "
        "adult rat cerebral cortex: a quantitative immunocytochemical "
        "description. J Comp Neurol 2000;428:305-318, PMID:11064369. Adult "
        "rat, ChAT immunocytochemistry: 'The mean density of ACh axons was "
        "the highest in the frontal cortex (13.0 m/mm(3) ...), as was the "
        "mean density of ACh varicosities (5.4 x 10(6)/mm(3) vs. 3.8 and "
        "4.6 x 10(6)/mm(3))' in somatosensory and visual cortex."
    ),
    "Parent2008": (
        "Parent M, Descarries L. Acetylcholine innervation of the adult rat "
        "thalamus. J Comp Neurol 2008;511:678-691, PMID:18924144. Same "
        "laboratory and the same ChAT method as [Mechawar2000], which is "
        "what makes the two comparable: 'The acetylcholine (ACh) "
        "innervation of thalamus arises mainly from the brainstem "
        "pedunculopontine and laterodorsal tegmental nuclei'; 'The regional "
        "density of ACh innervation was the highest in PF (2.1 x 10(6) "
        "varicosities/mm(3)), followed by Rt (1.7 x 10(6)) and DLG "
        "(1.3 x 10(6))'."
    ),
    "AllenCCF": (
        "Allen Mouse Brain Common Coordinate Framework structure graph 1, "
        "vendored at data/connectome/ and sliced into the committed fixture "
        "ontology. Every acronym below is a CCF acronym."
    ),
}


# ---------------------------------------------------------------------------
# What a nucleus is, and what a region is, in CCF acronyms
# ---------------------------------------------------------------------------

#: Each `projection_systems.NUCLEI` key, and the CCF structures whose
#: connectome ROWS are its projection. Eight of the nine are the nucleus
#: itself. TM is the exception and is not an editorial choice: the Knox et
#: al. 2019 region set has no row for the tuberomammillary nucleus as a
#: whole, and does have rows for both of its CCF children, so TM's
#: projection is the sum over TMd and TMv. Summing raw `connection_strength`
#: over source regions is exact (`volumes.py`: only the unnormalized matrix
#: is additive), which is why the derivation works from `strength` and not
#: from a published density.
NUCLEUS_STRUCTURES = {
    "VTA": ("VTA",),
    "SNc": ("SNc",),
    "DR": ("DR",),
    "CS": ("CS",),
    "LC": ("LC",),
    "NDB": ("NDB",),
    "PPN": ("PPN",),
    "LDT": ("LDT",),
    "TM": ("TMd", "TMv"),
}

#: The CCF structures of the medial hypothalamic zone that are NOT the
#: tuberomammillary nucleus, i.e. the terminal field a probe in the "medial
#: hypothalamus" of [Itoh1991] sampled. Listed explicitly rather than
#: resolved from the ontology at import time so that the committed table and
#: this module cannot disagree about what MEZ meant; a test re-derives the
#: list from the ontology and asserts it matches.
#:
#: The paper's probe placement cannot be resolved to a single CCF structure
#: from what it reports, so this is the whole zone minus the nucleus itself.
#: Including TMd and TMv instead would raise the reference density by
#: 1.27-fold, which a test measures and reports rather than assuming away.
MEZ_TERMINAL_FIELD = (
    "AHN", "LM", "MM", "SUM", "MPN", "PMd", "PMv", "PVHd", "VMH", "PH",
)

#: Model region name -> the CCF structures it is. Region names are the ones
#: `projection_scenarios.py` already uses; the CCF choices that are not a
#: literal name match follow P3.3's `connectome_scenarios.py` exactly rather
#: than making a second decision:
#:
#:   PFC -> PL   "The mouse CCF has no 'prefrontal cortex' structure. PL
#:               (prelimbic area) is the area most consistently treated as
#:               the rodent medial-prefrontal homologue". ILA and ACAd are
#:               the defensible alternatives and are in the table below so a
#:               test can re-run the map with each.
REGION_STRUCTURES = {
    "CP": ("CP",),
    "ACB": ("ACB",),
    "PFC": ("PL",),
    "PFC_ILA": ("ILA",),
    "PFC_ACAd": ("ACAd",),
    "MOp": ("MOp",),
    "HIP": ("CA1", "CA3", "DG"),
    "MD": ("MD",),
    "MEZ": MEZ_TERMINAL_FIELD,
    # The nuclear block: derived and recorded, never used. See the module
    # docstring.
    "VTA": ("VTA",),
    "SNc": ("SNc",),
    "DR": ("DR",),
    "CS": ("CS",),
    "LC": ("LC",),
    "NDB": ("NDB",),
    "PPN": ("PPN",),
    "LDT": ("LDT",),
    "TM": ("TMd", "TMv"),
}

#: The regions that are a source nucleus. A (nucleus, region) pair whose
#: region is in here is NUCLEAR: its value is recorded in the table as
#: evidence and is never used as a projection density.
NUCLEAR_REGIONS = frozenset(NUCLEUS_STRUCTURES)

#: Transmitter -> the region whose measured basal concentration
#: `projection_systems.CLEARANCE_BUILDERS` carries. See the module docstring.
REFERENCE_REGION = {
    "dopamine": "CP",
    "serotonin": "CP",
    "noradrenaline": "PFC",
    "acetylcholine": "CP",
    "histamine": "MEZ",
}

#: The provenance tags a `ProjectionTarget` built from this table carries.
DERIVED_PROVENANCE = "connectome:knox2019"
SOMATODENDRITIC_PROVENANCE = "illustrative-somatodendritic"

#: The committed derived table. Regenerate with
#: `scripts/make_projection_densities.py`, which reads only the
#: checksum-verified cache.
TABLE_PATH = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "projection_densities.csv"
)

#: Held-out measurements the derived map PREDICTS, used by
#: `validation_report()`. Each entry is
#: `(transmitter, numerator region, denominator region, low, high, source)`
#: where `[low, high]` is the measured ratio or the range the source
#: reports. Nothing is fitted to these; they are checks, and one of them
#: fails.
VALIDATION_RATIOS = (
    (
        "dopamine", "CP", "PFC", 13.0, 30.0,
        "[Ihalainen1999] MOUSE dialysate: 'DA in striatum was 13-30 times "
        "higher than in other areas'",
    ),
    (
        "dopamine", "CP", "PFC", 6.30, 6.30,
        "[Devoto2005] RAT, anaesthetized, matched 4 mm probes: caudate DA "
        "27.35 vs mPFC 4.34 pg/sample. The second, disagreeing measurement "
        "of the same ratio, kept because picking the agreeing one would be "
        "picking a result",
    ),
    (
        "serotonin", "CP", "PFC", 2.0, 2.0,
        "[Mathews2004] MOUSE zero net flux, same animals, same method: "
        "2.8 nM striatum vs 1.4 nM frontal cortex",
    ),
    (
        "noradrenaline", "PFC", "CP", 2.49, 2.49,
        "[Devoto2005] RAT, anaesthetized, matched 4 mm probes: mPFC NA 4.26 "
        "vs caudate 1.71 pg/sample",
    ),
    (
        "acetylcholine", "MD", "PFC", 0.24, 0.55,
        "[Parent2008] rat thalamus 1.3-2.1e6 ChAT varicosities/mm^3 against "
        "[Mechawar2000] rat cortex 3.8-5.4e6, same laboratory and the same "
        "ChAT method. A TERMINAL DENSITY rather than a concentration, which "
        "the model predicts with the same ratio because concentration is "
        "proportional to density at uniform uptake. Both regions' "
        "acetylcholine is projection-derived (thalamic from PPN/LDT, "
        "cortical from the basal forebrain), which is what makes the pair "
        "comparable at all; STRIATAL acetylcholine is not, so the much "
        "larger literature on striatal cholinergic terminals cannot be used "
        "here",
    ),
)

#: [Lin2023]'s measured histaminergic fibre density, as "% of pixels in this
#: CCF region that are labelled", for the regions the paper's own TEXT names
#: verbatim. THIS IS THE MODULE'S STRONGEST VALIDATION and it is worth
#: saying why in one place:
#:
#: * it is MOUSE, the species this whole project models;
#: * it is registered to Allen CCFv3, the SAME ontology the connectome rows
#:   are in, so no region has to be mapped by hand;
#: * it is CELL-TYPE SPECIFIC -- HDC-CreERT2 mice, so the labelled fibres
#:   are the histaminergic ones -- which is exactly the thing the derived
#:   map does NOT know (see "What the derived map does NOT know");
#: * and it is a whole-brain per-region table rather than a two-region
#:   ratio, so it tests the ORDERING over dozens of regions at once instead
#:   of one number.
#:
#: The 24 regions the paper lists as containing its AAV injection site (SI
#: Table S2: LM, TMd, TMv, PMd, PMv, TU, SUM, PSTN, MM, LHA, PH, PVp, IF,
#: DMH, VMH, IPN, PVi and the rest) carry the highest raw values and were
#: excluded from the authors' own grading; none of them is below, and
#: `fibre_density_rows()` also reports the correlation with every
#: hypothalamic region dropped, because the injection was in the
#: hypothalamus and a reader should not have to take the exclusion on trust.
HISTAMINE_FIBRE_DENSITY = {
    "NDB": 1.19, "SO": 0.87, "MS": 0.83, "VTA": 0.81, "SCH": 0.80,
    "MPN": 0.70, "AHN": 0.61, "EW": 0.61, "PVH": 0.54, "BST": 0.52,
    "SI": 0.50, "ARH": 0.47, "DR": 0.38, "SNc": 0.36, "CEA": 0.33,
    "PVT": 0.32, "ACB": 0.26, "SNr": 0.16, "MD": 0.15, "CLA": 0.11,
    "PB": 0.11, "LC": 0.09, "CP": 0.07, "LD": 0.07, "SUB": 0.07,
    "LGd": 0.06, "RT": 0.06, "VPM": 0.04, "MOB": 0.03, "CA1": 0.02,
    "CA3": 0.02, "NTS": 0.02,
}

SEARCHED_AND_EMPTY = """
**A QUANTITATIVE (zero- or no-net-flux) extracellular concentration in a
source nucleus, paired with one in its own terminal field.** That single
pair would replace this module's one illustrative density (the
somatodendritic one) with a measurement. The search found:

* [Tao2000] is the ONE within-study zero-net-flux pair that exists, and its
  terminal field is the nucleus accumbens rather than this model's
  caudoputamen reference; [Mateo2001] is within-animal but reports one
  range covering both regions; [Portas1998] is within-study but in
  fmol/sample. All three are recorded above as SUPPORT for the illustrative
  rule and none is used as its calibration.
* NO zero-net-flux or no-net-flux noradrenaline measurement in the locus
  coeruleus at all. Every locus coeruleus microdialysis study reached
  ([Pudovkina2001], [VanGaalen1997], [FernandezPastor2005], [Mateo2001],
  [Walling2004]) is conventional microdialysis, and three of the five report
  only percentage changes from their own baseline. A plausible reason is
  structural rather than accidental: a locus coeruleus probe has to be short
  and the noradrenaline in it sits near the assay's detection limit
  (20-25 fmol/sample in [Mateo2001]), so a no-net-flux regression there
  would be badly conditioned.
* Ratios assembled ACROSS laboratories are not usable, because the
  cross-laboratory spread on this quantity is itself of the size being
  measured: [PanLai1995]'s 1.59 nM and [Abercrombie1988]'s 14 nM for rodent
  noradrenaline differ 8.8-fold in two different terminal fields, which is
  larger than any nucleus:terminal ratio worth arguing about, and
  [Sacchetti1999] gets an 8-fold swing inside one region from probe geometry
  alone.

**A cell-type-specific whole-brain regionalized connectome for any of the
five transmitters.** Allen's Cre-line anterograde experiments exist; a
regionalized normalized model over them, of the kind Knox et al. 2019
published for wild type, was not found, and building one from the raw
experiments is a P3.1-sized data task rather than part of P6.3. The cost of
its absence is quantified rather than asserted: see the serotonin row of
`validation_report()`, which is the one prediction that fails, and the
"What the derived map does NOT know" section above for why cell type is the
leading suspect.
"""


# ---------------------------------------------------------------------------
# The derivation
# ---------------------------------------------------------------------------


def _structure_ids(ontology, acronyms):
    return [ontology.id_for_acronym(acronym) for acronym in acronyms]


def derive_density(strength, voxel_counts, ontology, source_acronyms,
                   target_acronyms, hemisphere="ipsi"):
    """`rho(source -> target)`, the module docstring's one formula.

    `strength` is a `connectome.RegionalConnectome` over the RAW
    `knox2019_connection_strength` matrix and `voxel_counts` is
    `volumes.target_voxel_counts(strength, density)`; passing an
    already-normalized matrix is refused, because summing per-size means is
    not the mean of the union (`volumes.py`).
    """
    if strength.dataset_key != "knox2019_connection_strength":
        raise ValueError(
            "a projection density is a sum of raw voxel-pair weights divided "
            "by a summed target volume, so it needs the unnormalized "
            "knox2019_connection_strength matrix; got %r"
            % (strength.dataset_key,)
        )
    block = strength.hemisphere(hemisphere)
    row_of = {int(i): k for k, i in enumerate(block.source_ids)}
    column_of = {
        int(i): k
        for k, i in zip(range(block.n_target), block.target_ids)
    }
    source_ids = _structure_ids(ontology, source_acronyms)
    target_ids = _structure_ids(ontology, target_acronyms)
    missing = [
        ontology[i].acronym
        for i in source_ids
        if i not in row_of
    ] + [
        ontology[i].acronym
        for i in target_ids
        if i not in column_of
    ]
    if missing:
        raise KeyError(
            "%s has no %s row/column for %s"
            % (strength.dataset_key, hemisphere, sorted(set(missing)))
        )
    total = 0.0
    for i in source_ids:
        for j in target_ids:
            total += float(block.weights[row_of[i], column_of[j]])
    volume = 0.0
    for j in target_ids:
        key = (hemisphere, int(j))
        if key not in voxel_counts:
            raise KeyError(
                "no %s voxel count for %s" % (hemisphere, ontology[j].acronym)
            )
        volume += float(voxel_counts[key])
    if volume <= 0.0:
        raise ValueError("target region has zero volume: %s" % (target_acronyms,))
    return total / volume


def derive_table(strength, voxel_counts, ontology, nuclei=None, regions=None):
    """The whole `{(nucleus, region): rho}` table, derived.

    `nuclei` and `regions` default to every key of `NUCLEUS_STRUCTURES` and
    `REGION_STRUCTURES`, which is what the committed table holds.
    """
    nuclei = list(NUCLEUS_STRUCTURES if nuclei is None else nuclei)
    regions = list(REGION_STRUCTURES if regions is None else regions)
    out = {}
    for nucleus in nuclei:
        for region in regions:
            out[(nucleus, region)] = derive_density(
                strength,
                voxel_counts,
                ontology,
                NUCLEUS_STRUCTURES[nucleus],
                REGION_STRUCTURES[region],
            )
    return out


# ---------------------------------------------------------------------------
# The table object
# ---------------------------------------------------------------------------


class DensityTable(object):
    """The derived densities, with the provenance of every number in them.

    One entry per (source nucleus, target region) pair. `kind` is
    `"terminal"` for a derived projection density and `"nuclear"` for a
    value that is recorded and refused; see the module docstring.
    """

    def __init__(self, densities, dataset_key, citation, units, source):
        self.densities = {
            (str(n), str(r)): float(v) for (n, r), v in dict(densities).items()
        }
        self.dataset_key = dataset_key
        self.citation = citation
        self.units = units
        self.source = source

    # --- reads -----------------------------------------------------------

    @property
    def nuclei(self):
        return sorted({n for n, _ in self.densities})

    @property
    def regions(self):
        return sorted({r for _, r in self.densities})

    def kind(self, region):
        return "nuclear" if region in NUCLEAR_REGIONS else "terminal"

    def density(self, region, nucleus):
        """`rho(nucleus -> region)` as derived, whatever its kind."""
        try:
            return self.densities[(nucleus, region)]
        except KeyError:
            raise KeyError(
                "no derived density for %s -> %s; the table has nuclei %s and "
                "regions %s" % (nucleus, region, self.nuclei, self.regions)
            )

    def total(self, region, nuclei):
        return sum(self.density(region, nucleus) for nucleus in nuclei)

    def relative(self, region, reference, nuclei):
        """A region's total density as a multiple of the reference's."""
        return self.total(region, nuclei) / self.total(reference, nuclei)

    # --- the map a projection system consumes ----------------------------

    def target_densities(self, regions, nuclei, reference, nodes=None):
        """`{region: {node: rho}}` for one projection system.

        Terminal regions get the derived density. A NUCLEAR region gets the
        reference region's total density from its own nucleus and zero from
        every other, which is the one illustrative number in this module;
        see the module docstring for both reasons it is not derived.

        `nodes` maps a nucleus key to the circuit node name the scenario
        gives it, defaulting to the nucleus key itself.
        """
        nodes = dict(nodes or {})
        nuclei = list(nuclei)
        somatodendritic = self.total(reference, nuclei)
        out = {}
        for region in regions:
            entry = {}
            for nucleus in nuclei:
                node = nodes.get(nucleus, nucleus)
                if region in NUCLEAR_REGIONS:
                    entry[node] = (
                        somatodendritic if region == nucleus else 0.0
                    )
                else:
                    entry[node] = self.density(region, nucleus)
            out[region] = entry
        return out

    def provenance(self, region):
        return (
            SOMATODENDRITIC_PROVENANCE
            if region in NUCLEAR_REGIONS
            else DERIVED_PROVENANCE
        )

    # --- reports ---------------------------------------------------------

    def report(self, nuclei=None, regions=None):
        """The whole table as text, terminal regions first."""
        nuclei = self.nuclei if nuclei is None else list(nuclei)
        regions = self.regions if regions is None else list(regions)
        ordered = [r for r in regions if self.kind(r) == "terminal"] + [
            r for r in regions if self.kind(r) == "nuclear"
        ]
        lines = [
            "Projection densities derived from %s" % self.dataset_key,
            "=" * 72,
            "units: %s" % self.units,
            "source: %s" % self.source,
            "",
            "%-6s %-9s %s" % ("region", "kind", "".join("%11s" % n for n in nuclei)),
        ]
        for region in ordered:
            lines.append(
                "%-6s %-9s %s"
                % (
                    region,
                    self.kind(region),
                    "".join("%11.4g" % self.density(region, n) for n in nuclei),
                )
            )
        lines.append("")
        lines.append(
            "NUCLEAR rows are recorded and REFUSED: a nucleus's own "
            "extracellular pool is"
        )
        lines.append(
            "somatodendritic, which no anterograde tracer measures. See the "
            "module docstring."
        )
        return "\n".join(lines)

    def validation_report(self):
        """Every prediction in `VALIDATION_RATIOS`, measured against its
        source. Reports the error; nothing here is fitted and one row fails.
        """
        from . import projection_systems as _ps

        lines = [
            "Derived-density validation: predicted concentration ratios "
            "against held-out measurements",
            "=" * 78,
            "",
            "%-14s %-11s %10s %14s %9s %10s" % (
                "transmitter", "ratio", "predicted", "measured", "error",
                "direction",
            ),
        ]
        for row in self.validation_rows():
            lines.append(
                "%-14s %-11s %10.3g %14s %9s %10s"
                % (
                    row["transmitter"],
                    "%s:%s" % (row["numerator"], row["denominator"]),
                    row["predicted"],
                    row["measured"],
                    row["verdict"],
                    "same" if row["same_direction"] else "OPPOSITE",
                )
            )
        lines.append("")
        for row in self.validation_rows():
            lines.append("  %-14s %s" % (row["transmitter"], row["source"]))
        return "\n".join(lines)

    def validation_rows(self):
        """`VALIDATION_RATIOS` evaluated, as dicts, so a test asserts on
        values rather than on a string."""
        from . import projection_systems as _ps

        rows = []
        for transmitter, numerator, denominator, low, high, source in (
            VALIDATION_RATIOS
        ):
            nuclei = _ps.NUCLEI_BY_TRANSMITTER[transmitter]
            predicted = self.total(numerator, nuclei) / self.total(
                denominator, nuclei
            )
            if low <= predicted <= high:
                error, verdict = 1.0, "in range"
            elif predicted < low:
                error, verdict = low / predicted, "%.2gx LOW" % (low / predicted)
            else:
                error, verdict = predicted / high, "%.2gx HIGH" % (predicted / high)
            rows.append(
                {
                    "transmitter": transmitter,
                    "numerator": numerator,
                    "denominator": denominator,
                    "predicted": predicted,
                    "low": low,
                    "high": high,
                    "measured": (
                        "%.3g" % low if low == high else "%.3g-%.3g" % (low, high)
                    ),
                    "error": error,
                    "verdict": verdict,
                    # Does the model at least get which region is richer?
                    "same_direction": (predicted - 1.0) * (0.5 * (low + high) - 1.0)
                    > 0.0,
                    "source": source,
                }
            )
        return rows

    def __repr__(self):
        return "DensityTable(%d nuclei x %d regions, source=%r)" % (
            len(self.nuclei),
            len(self.regions),
            self.source,
        )


# ---------------------------------------------------------------------------
# The committed table
# ---------------------------------------------------------------------------

TABLE_UNITS = (
    "connection density: summed voxel-pair model weight from the source "
    "nucleus divided by the target region's voxel count (Knox et al. 2019 "
    "'connection_density', computed at this grouping); dimensionless, and "
    "only RATIOS of it are used"
)


def write_table(path, densities, dataset_key, citation, units, header_lines=()):
    """Write a derived table to CSV. Used by
    `scripts/make_projection_densities.py` and by nothing else."""
    with open(path, "w", newline="") as handle:
        for line in header_lines:
            handle.write("# %s\n" % line)
        writer = csv.writer(handle)
        writer.writerow(["nucleus", "region", "kind", "structures", "density"])
        for nucleus in NUCLEUS_STRUCTURES:
            for region in REGION_STRUCTURES:
                key = (nucleus, region)
                if key not in densities:
                    continue
                writer.writerow(
                    [
                        nucleus,
                        region,
                        "nuclear" if region in NUCLEAR_REGIONS else "terminal",
                        "+".join(REGION_STRUCTURES[region]),
                        repr(float(densities[key])),
                    ]
                )
    return path


def load_table(path=None):
    """The committed derived table.

    Real derived data, committed for the same reason the connectome
    FIXTURES are: the offline gate cannot read the gitignored 8 MB cache,
    and a table of 162 numbers can be reviewed in a diff. It is regenerated
    from the checksum-verified cache by
    `scripts/make_projection_densities.py`, `tests/test_projection_density.py`
    re-derives every pair the committed 15-region connectome fixture carries
    and asserts they match exactly, and a `full_data_only` test re-derives
    the WHOLE table from the real matrices and asserts the same.
    """
    path = TABLE_PATH if path is None else path
    densities = {}
    with open(path, "rt", newline="") as handle:
        rows = [line for line in handle if not line.startswith("#")]
    reader = csv.DictReader(rows)
    for row in reader:
        nucleus, region = row["nucleus"], row["region"]
        expected = "+".join(REGION_STRUCTURES[region])
        if row["structures"] != expected:
            raise ValueError(
                "%s: region %r is %s in the committed table and %s in "
                "REGION_STRUCTURES; the table is stale"
                % (path, region, row["structures"], expected)
            )
        densities[(nucleus, region)] = float(row["density"])
    return DensityTable(
        densities=densities,
        dataset_key="knox2019_connection_strength",
        citation=CITATIONS["Knox2019"],
        units=TABLE_UNITS,
        source="committed table (%s)" % os.path.basename(path),
    )


def load_cache_table(cache_dir=None, allow_download=False):
    """Re-derive the table from the checksum-verified cache. Needs the
    gitignored `data/connectome/`; `load_table()` is the offline path."""
    from .connectome import load_knox_inputs, load_ontology

    ontology = load_ontology(cache_dir=cache_dir, allow_download=allow_download)
    strength, voxels = load_knox_inputs(
        cache_dir=cache_dir, ontology=ontology, allow_download=allow_download
    )
    return DensityTable(
        densities=derive_table(strength, voxels, ontology),
        dataset_key=strength.dataset_key,
        citation=strength.citation,
        units=TABLE_UNITS,
        source="derived from the verified cache",
    )


def uniform_table(value=1.0):
    """P6.1's map: 1.0 from every nucleus into every region.

    Kept as an object rather than as a memory, so that every result P6.3
    changes can be re-run at P6.1's densities and the change REPORTED
    instead of asserted. Nothing in the modelling path builds one.
    """
    return DensityTable(
        densities={
            (n, r): float(value)
            for n in NUCLEUS_STRUCTURES
            for r in REGION_STRUCTURES
        },
        dataset_key="none",
        citation="P6.1's illustrative-uniform map; not data",
        units="ILLUSTRATIVE uniform density, dimensionless",
        source="uniform (P6.1)",
    )


def fibre_density_rows(strength=None, voxel_counts=None, ontology=None,
                       exclude_hypothalamus=False):
    """[Lin2023]'s measured histaminergic fibre density against the derived
    tuberomammillary projection density, region by region.

    Needs the CONNECTOME rather than the committed table, because most of
    [Lin2023]'s regions are not in `REGION_STRUCTURES` (they are not regions
    this model has a node or a receptor for). That is the point of the
    check: it tests the derivation over 32 regions chosen by somebody else.

    Returns `(rows, spearman, log_pearson)` where each row is
    `(acronym, derived, measured)`.
    """
    from scipy import stats

    from .connectome import load_knox_inputs, load_ontology

    if ontology is None:
        ontology = load_ontology()
    if strength is None or voxel_counts is None:
        strength, voxel_counts = load_knox_inputs(ontology=ontology)
    hypothalamus = ontology.id_for_acronym("HY")
    rows = []
    for acronym, measured in sorted(HISTAMINE_FIBRE_DENSITY.items()):
        structure = ontology.id_for_acronym(acronym)
        if exclude_hypothalamus and (
            structure == hypothalamus
            or ontology.is_descendant_of(structure, hypothalamus)
        ):
            continue
        rows.append(
            (
                acronym,
                derive_density(
                    strength,
                    voxel_counts,
                    ontology,
                    NUCLEUS_STRUCTURES["TM"],
                    (acronym,),
                ),
                float(measured),
            )
        )
    derived = np.array([row[1] for row in rows])
    measured = np.array([row[2] for row in rows])
    return (
        rows,
        float(stats.spearmanr(derived, measured).statistic),
        float(stats.pearsonr(np.log(derived), np.log(measured)).statistic),
    )


def fibre_density_report(strength=None, voxel_counts=None, ontology=None):
    """The [Lin2023] comparison as text. Needs the gitignored cache."""
    lines = [
        "Derived tuberomammillary projection density against [Lin2023]'s "
        "measured histaminergic fibre density",
        "=" * 78,
        "",
        "Mouse, cell-type-specific (HDC-CreERT2), whole-brain, and already in "
        "Allen CCFv3, so no",
        "region is mapped by hand. This is the only check in this module "
        "that tests the ORDERING",
        "over dozens of regions rather than one ratio.",
        "",
    ]
    rows, spearman, log_pearson = fibre_density_rows(
        strength, voxel_counts, ontology
    )
    _, spearman_ex, _ = fibre_density_rows(
        strength, voxel_counts, ontology, exclude_hypothalamus=True
    )
    lines.append("%-6s %14s %14s" % ("region", "derived rho", "Lin2023 %"))
    for acronym, derived, measured in sorted(rows, key=lambda r: -r[2]):
        lines.append("%-6s %14.5g %14.2f" % (acronym, derived, measured))
    lines.append("")
    lines.append(
        "Spearman rank correlation over %d regions: %.4f" % (len(rows), spearman)
    )
    lines.append("Pearson correlation of the logs:          %.4f" % log_pearson)
    lines.append(
        "Same, with every hypothalamic region dropped (the injection was in "
        "the"
    )
    lines.append(
        "hypothalamus, and the reader should not have to take the authors' "
        "own"
    )
    lines.append("injection-site exclusion on trust): %.4f" % spearman_ex)
    return "\n".join(lines)


def concentration_report(table=None):
    """Every region's predicted basal concentration, per transmitter.

    The reference region's value is its measurement; every other terminal
    region's is a prediction of the density map, and every nuclear region's
    carries the one illustrative density. Written to `results/` by
    `scripts/report_projection_densities.py`.
    """
    from . import projection as _projection
    from . import projection_systems as _ps

    table = load_table() if table is None else table
    lines = [
        "Predicted basal extracellular concentrations from the derived "
        "density map",
        "=" * 78,
        "",
    ]
    for transmitter in ("dopamine", "serotonin", "noradrenaline",
                        "acetylcholine", "histamine"):
        nuclei = _ps.NUCLEI_BY_TRANSMITTER[transmitter]
        reference = REFERENCE_REGION[transmitter]
        basal = _ps.CLEARANCE_BUILDERS[transmitter]().basal
        lines.append(
            "%s: nuclei %s, reference %s = %.3g nM (MEASURED)"
            % (transmitter, ", ".join(nuclei), reference, basal)
        )
        clearance = _ps.CLEARANCE_BUILDERS[transmitter]()
        release_scale = clearance.basal_release_rate() / table.total(
            reference, nuclei
        )
        for region in table.regions:
            if table.kind(region) != "terminal":
                continue
            relative = table.relative(region, reference, nuclei)
            # The exact inversion of the uptake law, not `relative * basal`.
            # Michaelis-Menten clearance is CONVEX in the release rate, so a
            # region above the reference sits at MORE than proportionally
            # more; `projection.py`'s docstring puts the departure at 1-2%
            # near the measured basal levels, and it is larger than that for
            # the regions this map puts far above one. A region whose demand
            # exceeds Vmax has no steady state at all, which is the same
            # refusal the nuclear block gets and is reported as such.
            try:
                predicted = _projection.concentration_for_uptake(
                    clearance, release_scale * table.total(region, nuclei)
                )
                value = "%10.4g nM" % predicted
            except ValueError:
                value = " no steady state (release exceeds Vmax)"
            lines.append(
                "    %-8s %8.4g x reference   %s   %s"
                % (
                    region,
                    relative,
                    value,
                    "MEASURED" if region == reference else "predicted",
                )
            )
        lines.append("")
    lines.append(
        "Nuclear regions are held at the reference concentration by the one "
        "illustrative"
    )
    lines.append(
        "density in this module (`illustrative-somatodendritic`), which is "
        "P6.1's value."
    )
    return "\n".join(lines)


__all__ = [
    "CITATIONS",
    "DERIVED_PROVENANCE",
    "MEZ_TERMINAL_FIELD",
    "NUCLEAR_REGIONS",
    "NUCLEUS_STRUCTURES",
    "REFERENCE_REGION",
    "REGION_STRUCTURES",
    "SEARCHED_AND_EMPTY",
    "SOMATODENDRITIC_PROVENANCE",
    "TABLE_PATH",
    "TABLE_UNITS",
    "VALIDATION_RATIOS",
    "HISTAMINE_FIBRE_DENSITY",
    "DensityTable",
    "concentration_report",
    "derive_density",
    "derive_table",
    "fibre_density_report",
    "fibre_density_rows",
    "load_cache_table",
    "load_table",
    "uniform_table",
    "write_table",
    "validation_report",
]


def validation_report(table=None):
    """Module-level shortcut for `DensityTable.validation_report`."""
    return (load_table() if table is None else table).validation_report()
