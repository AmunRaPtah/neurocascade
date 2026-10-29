# Pre-registration: Phase 35 GO / NO-GO on SIDER-powered liability labels

Written and sealed 2026-10-29, before any SIDER label is built onto the drug
universe and before any candidate is selected on it. It follows the route
`docs/prereg-p33-4-go.md` ("One look") sets after a NO-GO / no-verdict: a new
task, a new pre-registration, and a held-out set whose gating labels have not
been used. The user asked on 2026-10-29 to keep grinding to the full product
goal rather than narrow scope; this is the human decision Phase 34's
no-verdict waited for. P32.x and P30.4 are gated on **P35.4** and still need a
GO.

## Why a new label source

Every prior phase ended underpowered. The strict placebo-controlled label
tables (openFDA, EU SmPC) reach about 15-19 of 334 universe drugs per
liability; EPS has 8 positives and orthostatic hypotension 3 across the whole
v2 universe. P34.4, P34.4b and P34.6 showed the approved-drug universe is
exhausted under that label rule. SIDER carries placebo-controlled side-effect
frequencies for 1,430 drugs (`/root/datalake/tables/clinical/sider_frequencies.parquet`,
291,632 rows, `placebo_flag` and `frequency_lower`/`frequency_upper` columns).
Term coverage across SIDER drugs: somnolence 668, dry mouth 558, constipation
703, extrapyramidal 102, akathisia 69, orthostatic 254, seizure 50. This is
the **same strict placebo-controlled label idea at 10-40x the drugs**, not a
weaker rule and not a scope change.

## Label source and definition (fixed now, applied mechanically by P35.2/P35.4)

- **Source:** `sider_frequencies` rows. STITCH compound ids map to drugs by
  PubChem CID (the STITCH `CID1...`/`CID0...` flat id, stripped to its integer
  CID, matched to each universe drug's PubChem CID resolved from its ChEMBL id;
  the resolution table is built once in P35.2 and committed).
- **Positive** for a liability: the drug has at least one mapped MedDRA PT whose
  SIDER frequency (`frequency_upper`, or the parsed `frequency_desc` when the
  numeric bound is null) is **>= 1%**. **Negative:** the drug has a SIDER record
  (appears in `sider_frequencies` or `sider_side_effects`) but no mapped PT for
  that liability at >= 1%. **Excluded (not negative):** a drug with no SIDER
  record at all. The single 1% boundary (T_pos = T_neg = 1%) is fixed here.
- **Placebo handling:** where a placebo-flagged row exists for the same
  drug/PT, the placebo frequency is subtracted before the 1% test; where it
  does not, the raw frequency is used and the drug/PT is flagged
  `placebo_uncontrolled` in the build output. The flag is reported, not used to
  drop the row.
- **MedDRA PT -> liability map (fixed now):**
  - sedation: Somnolence, Sedation, Hypersomnia
  - anticholinergic: Dry mouth, Constipation, Urinary retention, Blurred vision,
    Mydriasis
  - eps: Extrapyramidal disorder, Akathisia, Dystonia, Parkinsonism, Tardive
    dyskinesia, Tremor, Muscle rigidity
  - orthostatic_hypotension: Orthostatic hypotension, Postural dizziness
  - seizure: Seizure, Convulsion, Epilepsy, Grand mal convulsion
  Matching is on the SIDER `side_effect_name` by exact PT string (case-folded);
  every matched and unmatched PT is listed in the build output.
- **FAERS / OFFSIDES** (`/root/datalake/tables/clinical/offsides.parquet`) is a
  **disclosed secondary** reported for calibration only. It never sets a gating
  label.

## Drug universe and split

- **Universe:** a drug is in the universe iff it has (a) a measured Ki at >= 1
  P33.1 anchor target from `scripts/v2_binding.py` unchanged (ChEMBL, GtoPdb,
  PDSP), (b) a SIDER record, and (c) a Cmax from `scripts/v2_cmax.py` unchanged.
  Every universe member is listed; the three input sources are frozen as of
  P35.2.
- **Split:** deterministic by `sha256(inn) mod 10`: held-out iff the digit is in
  {0, 1} (a ~20% held-out fraction), train otherwise. The rule is fixed here and
  computed by `guards` the same way `v2_heldout` is; held-out membership is not
  stored, it is recomputed. `guards.hide_heldout` strips the held-out labels
  from every session tree and a test asserts the candidate path imports no
  held-out label.

## Order (the anti-tuning rule)

1. P35.2 builds **train** labels, the CID map and the universe. The held-out
   labels are not built into the session tree.
2. P35.3 selects and **freezes** the candidate on train only: code and fitted
   parameters committed, their sha256 in `results/P35.3.json`, before any
   held-out label exists in the tree. The candidate may use the extra train
   power (re-selection is allowed on train); it may not be one of the baselines.
3. P35.4 builds the held-out labels and scores the frozen candidate **once**,
   host-side, under `NC_HOST_EVAL`. The first `NC_GO_VERDICT` is written to
   `HOME/go/P35.4.json` and binds.

## Rule

- **Statistic and decision:** `pipeline/go_verdict.py`, **unchanged**. Candidate
  AUROC minus best-baseline AUROC on the common set, paired bootstrap of drugs
  (2000 resamples, seed 20261026, percentile 95% interval). **GO** if the
  interval lies entirely above 0 for at least one liability whose held-out
  common set has >= 8 positives and >= 8 negatives. **NO-GO** otherwise.
- **Baselines:** strongest-target Ki, 1-NN fingerprint (fitted on train only),
  ATC-class prevalence (train prevalence), plasma Cmax,u/Ki occupancy
  (occupancy_max). The candidate may not be any of these.
- **Secondary, pre-committed:** the already-frozen P34.3 candidate
  (`results/P34.3.json`, family `candidate_rank_combined`, sha256 `d94aaecab870`)
  is scored on the same held-out set and reported beside the P35.3 candidate.
  It is a secondary read, not the gating statistic.
- **NO-GO** holds P32.x and P30.4 for the human. No Phase 36 retry is started
  without them.

## Secondary, not gating

Human CL and Vss fold error against the train mean, as in P33.4.

## Disclosed weaknesses

- SIDER label-rate is noisier than a hand-curated placebo-controlled incidence;
  where no placebo-flagged row exists the raw frequency is used (flagged).
- The PT -> liability map and the 1% boundary are judgements fixed in advance,
  not optimised.
- Universe drugs overlap the v2 universe in pharmacology, so this tests transfer
  to a larger labelled set under a new label source, not to new mechanisms.
