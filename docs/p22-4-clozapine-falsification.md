# P22.4: the clozapine anomaly as a falsification test

**Pre-registered in `docs/p22-preregistration.md` (P22.4). The outcome→reading
mapping was fixed before any number here was computed. Nothing is tuned to make
clozapine fit, in either direction (C1).**

Clozapine is the most effective antipsychotic available and is clinically
effective at striatal D2 occupancy well below the 65% response threshold this
repository sources from Kapur 2000. The test asks whether the untuned chain
reproduces that real anomaly. It does.

## Result: Outcome A — anomaly reproduced, hypothesis falsified

Under the primary registered constant (Gründer 2006 putamen EC50 950 ng/mL, Emax
approached 100%, recorded 1.0), the model's own hyperbola requires:

| Target striatal occupancy | Plasma clozapine needed |
|---|---|
| 0.65 (response, Kapur 2000) | **1764 ng/mL** |
| 0.78 (EPS, Kapur 2000) | **3368 ng/mL** |

The clinically-effective plasma range for clozapine is **350–600 ng/mL** (AGNP
consensus, Hiemke 2018, PMID 28910830). So the concentration the model says is
needed for 65% striatal occupancy sits **2.94× above the top of the range at
which clozapine actually works**, and 5.0× above its lower response threshold.

Turned around: across the clinically-effective plasma range, the model predicts
only **27–39% striatal D2 occupancy** — which matches what Gründer *measured*
(putamen 36%, caudate 43%). The chain reproduces the anomaly quantitatively, not
just directionally.

**Reading (fixed in advance): this is Outcome A.** The untuned chain places
clozapine's 65% concentration far above the concentrations at which clozapine is
clinically effective. That is a model success and a hypothesis failure at once:
the model correctly reproduces the real clozapine anomaly, and in doing so it
**falsifies occupancy-threshold (0.65–0.78 striatal D2) as the sole mediator of
antipsychotic response** for clozapine. The hypothesis cannot be both true and
reconciled with clozapine's efficacy at 350–600 ng/mL.

## Robustness across every registered candidate (none chosen after the fact)

| Region / fit | EC50 (ng/mL) | Emax | C65 (ng/mL) | vs range top (600) | Outcome |
|---|---|---|---|---|---|
| **putamen (primary)** | 950 | 1.0 | 1764 | 2.94× | **A** |
| caudate | 582 | 1.0 | 1081 | 1.80× | A |
| Uchida fixed-Emax | 483 | 1.0 | 897 | 1.50× | A |
| Uchida free-Emax | 105 | 0.60 | **unreachable** | — | A (strongest) |
| inferior temporal cortex *(extrastriatal)* | 333 | 1.0 | 618 | 1.03× | A, sensitivity only |

Every striatal candidate gives Outcome A. The Uchida free-Emax fit (ceiling 60%)
puts 65% occupancy out of reach at *any* plasma concentration — the strongest
form of the anomaly. The one candidate that lands *near* the threshold rather
than far above it is the **extrastriatal** cortex EC50 (C65 618 ng/mL, just over
the range top), which is exactly Gründer's own thesis — clozapine binds cortical
D2/D3 preferentially, and "cortical occupancy approaches 60% ... in the range
350–400 ng/ml, which corresponds to the threshold for antipsychotic efficacy."
Because the 0.65/0.78 thresholds were measured striatally, the cortical row is
reported as sensitivity and never decides the headline.

## Corroboration without a PK model: the label's own concentrations

The pinned VERSACLOZ snapshot (NDA203479) states that across 100–800 mg/day once
daily, the average steady-state peak plasma was 275 ng/mL (range 105–723), trough
75 ng/mL (range 11–198). Even the **highest observed peak** across the whole
label dose range, 723 ng/mL, gives only **43% striatal occupancy** under the
putamen fit — still below 65%. The anomaly holds against the label's measured
concentrations, not only against the consensus range.

## Deviation from the pre-registration (logged under C1)

The pre-registration's literal P22.4 procedure computes **D65 as a dose** and the
occupancy at the VERSACLOZ label dose range (300–900 mg/day). That needs a
clozapine dose→plasma PK bridge, and the chain does not have one:

- P21.3 recorded clozapine population-PK as **searched-and-empty** — VERSACLOZ
  NDA203479 performed no popPK analysis, and the older applications predate
  electronic review posting.
- No clozapine PK set is hand-transcribed in `pk_parameters`, and
  `fda_pk_bridge` has nothing to convert.
- P22.2 (ledger row V8) already recorded the same pk-layer refusal.

So the dose-domain test is reported as a **pk-layer refusal** (commitment C2, no
silent drops; see `clozapine_falsification.dose_domain_refusal`), and the
falsification is run in the **concentration domain**, which needs no PK bridge.

This is not a workaround — it is the sharper form of the test. The clozapine
anomaly is a statement about plasma concentration and occupancy (Nordström 1995,
Gründer 2006, Kapur 2000 all read it there), and clozapine is dosed clinically to
a plasma target precisely because dose→plasma varies >45-fold between patients
(Potkin 1994). Comparing the model's 65%-occupancy *concentration* against the
clinically-effective *concentration* range removes the one unsourced link
(dose→plasma) rather than inventing it. The pre-registration's A/B logic
translates exactly: Outcome A if C65 exceeds the top of the effective range or
does not exist; Outcome B otherwise; a tie resolves to B.

## Why clozapine's EC50 is not in the live registry

It is held in `neurocascade.human.clozapine_falsification` as schema-validated
`D2PlasmaEC50` records (ingested under P22.2/P22.3's rules — the same dataclass
and its validation), **not** inserted into `D2_PLASMA_EC50`. Registering it live
would retroactively change the committed P22.2 (V8) and P22.3 clozapine records,
which recorded clozapine as refused / searched-and-empty before this task ran;
C1 forbids changing anything affecting a drug after its comparison was seen.
clozapine is additionally the suite's canonical "unknown drug" fixture for the
refusal-layer tests. The records are therefore kept outside the chain.

## Ground-truth sources (none chosen in this repository)

- **AGNP consensus**: Hiemke C, et al. Pharmacopsychiatry 2018;51(1-02):9-62.
  PMID 28910830. Clozapine therapeutic reference range 350–600 ng/mL. *(primary)*
- Potkin SG, et al. J Clin Psychiatry 1994;55 Suppl B:133-6. PMID 7961557.
  420 ng/mL optimally separates responders (60% above) from non-responders (8%
  below) at a fixed 400 mg/day.
- VanderZwaag C, et al. Am J Psychiatry 1996;153(12):1579-84. PMID 8942454.
  Response best in the high (350–450 ng/mL) of three randomised serum ranges.
- Oyewumi LK, et al. Ther Drug Monit 1995;17(2):137-41. PMID 7624901. Putative
  threshold 350 µg/L.
- VERSACLOZ label, NDA203479, pinned snapshot (`data/fda/`, SHA-256 in
  `fda_sources`).

EC50 source: Gründer G, et al. Neuropsychopharmacology 2006;31(5):1027-35. PMID
16237387 (putamen, cortex printed in the abstract; caudate 582 ng/mL and the
Uchida 2011 pooled values via the open secondary Spangemacher 2025, PMC12488982).
Clozapine molar mass 326.82 g/mol from the Clozapine USP ODT label (ANDA090308).
