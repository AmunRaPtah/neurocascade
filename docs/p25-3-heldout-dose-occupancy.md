# P25.3: oral dose to D2 occupancy, end to end, for the drugs with accepted PK

Every number below comes from `neurocascade.human.heldout_dose_occupancy`, and
`tests/test_heldout_dose_occupancy.py` pins each one. The ledger row is V15.

## Result

**Four drugs now run from an oral dose to striatal D2 occupancy, and each one is
compared against PET data from a group that did not supply its constant. All
four are within the pre-registered P22.3 threshold of MAE ≤ 0.10.** Haloperidol
is V4, recomputed and not re-scored. The three new comparisons:

| drug | PK (P25.1/P25.2) | occupancy constant | held-out PET | n obs / groups | MAE | bias | |
|---|---|---|---|---|---|---|---|
| haloperidol | Pilla Reddy 2013 | P19.4 fit | Kapur 1996, 2 mg/day, range 0.53–0.74 | range | steady-state mean 0.5635 contained | – | V4, not blind |
| aripiprazole | Knights 2015 (74.19 kg) | Kim 2012, 11.1 ng/mL | Ito 2012, single 3/6/9 mg, scan 3.5–4.5 h | 6 / 3 | 0.0406 | −0.0406 | pass |
| quetiapine | Zheng 2024 (70 kg) | Nord 2011 IR, 1373 nM | Tauscher-Wisniewski 2002, once daily 427 mg, 2 h and 20 h | 2 / 1 | 0.0597 | −0.0597 | pass |
| olanzapine | LYBALVI bridge (70 kg) | de Greef 2011, 5.29 ng/mL, Emax 0.875 | Mamo 2008 oral baseline, 15.2 mg/day, trough | 1 / 1 | 0.0584 | +0.0584 | pass |

Per observation:

| drug | group | scan (h after dose) | region | predicted plasma (ng/mL) | observed plasma | predicted | observed ± SD | error |
|---|---|---|---|---|---|---|---|---|
| aripiprazole | 3 mg, n=5 | 3.5–4.5 | caudate | 11.12 | 12.0 | 0.5005 | 0.57 ± 0.04 | −0.0695 |
| aripiprazole | 3 mg, n=5 | 3.5–4.5 | putamen | 11.12 | 12.0 | 0.5005 | 0.55 ± 0.02 | −0.0495 |
| aripiprazole | 6 mg, n=5 | 3.5–4.5 | caudate | 22.25 | 29.0 | 0.6671 | 0.73 ± 0.03 | −0.0629 |
| aripiprazole | 6 mg, n=5 | 3.5–4.5 | putamen | 22.25 | 29.0 | 0.6671 | 0.69 ± 0.03 | −0.0229 |
| aripiprazole | 9 mg, n=2 | 3.5–4.5 | caudate | 33.37 | 40.0 | 0.7504 | 0.78 | −0.0296 |
| aripiprazole | 9 mg, n=2 | 3.5–4.5 | putamen | 33.37 | 40.0 | 0.7504 | 0.76 | −0.0096 |
| quetiapine | 427 mg qd | 2 | striatum | 644.4 | – | 0.5503 | 0.62 ± 0.10 | −0.0697 |
| quetiapine | 427 mg qd | 20 | striatum | 52.2 | – | 0.0903 | 0.14 ± 0.08 | −0.0497 |
| olanzapine | 15.2 mg qd, n=14 | 24 (trough) | striatum | 31.55 | 37.4 ± 31.2 | 0.7494 | 0.691 ± 0.152 | +0.0584 |

**The plasma arm is reported but decides nothing.** Wherever the PET source
printed plasma, the prediction is low by a factor of 0.77–0.93, and every ratio
is inside the V5/V13 twofold threshold.

## The brief was wrong about structure (rule 11)

The brief asked for "the missing brain-exposure (Kp,uu) and D2 affinity
pieces". **This chain has no Kp,uu step and no in-vitro affinity step, and
adding either would be an error.** Since P19.5 the chain has driven occupancy
with a plasma EC50 fitted in living people. That constant already contains brain
penetration, plasma protein binding and competition with endogenous dopamine.
Multiplying by a Kp,uu, or swapping in a Ki, would count those effects twice.

So "filling in" the two pieces means two things for each new drug:

1. **A first-hand plasma EC50 from human [11C]raclopride PET.**
   - Aripiprazole: Kim 2012, the same row P22.3 used.
   - Quetiapine: Nord 2011's IR-only K_i,app, 1373 nM. The pooled IR+XR value
     is 1369 nM and the XR-only value is 1365 nM. IR was chosen because the PK
     record and the test are both IR, and the choice moves the constant by 0.3%.
2. **A Kp,uu record with status `not-required` and no value**, naming that
   EC50. The chain's double-counting guard reads it.

Neither drug has a measured human Kp,uu anywhere, and none would be used if it
did.

**Where these rows live.** They are kept in the P25.3 module and passed to
`predict_occupancy` explicitly. They are not added to `D2_PLASMA_EC50` or
`HUMAN_KP_UU_RECORDS`, for P22.3's reason (deviation D1): those registries drive
the frozen P22.2 and P22.5 results and the public interface's refusal lists. To
make this possible, `predict_occupancy` gained a `kp_uu_record` argument, which
mirrors the existing `pk_record` and `receptor_record` and is checked exactly
like the registry's record.

## The one gate that was opened, and why it is narrow

Aripiprazole has been refused at the active-metabolite layer since P21.4. That
refusal is a regimen mismatch:
- Kim 2012 fitted its parent-only EC50 after a **single** oral dose;
- the label's 40% dehydro-aripiprazole share is a **steady-state** figure.

Ito 2012 measured the metabolite in the test subjects themselves, during the
scan: 0.4 against 12.0 ng/mL, 0.9 against 29.0 and 1.1–1.6 against 39.6–40.4.
That is **about 3% of the parent**. So in the single-dose regimen, the reason for
refusal does not apply.

The change is narrow:
- `ActiveMetabolite` gains `parent_only_regimen` and the measurement that
  justifies it. A record may have one only if it is `refused` and the evidence
  is filled in.
- `require_parent_only_path(..., regimen=)` clears a record only when the
  caller declares exactly that regimen. `predict_occupancy` forwards its new
  `regimen` argument to the gate.
- `steady_state_occupancy` refuses any `regimen` argument, because it
  superposes the single-dose curve into a repeated-dose curve.
- Risperidone's refusal is about which axis exists, not about regimen, so it
  gets no clearing regimen.
- Without the argument, every existing caller behaves exactly as before, and
  `predict()` still refuses aripiprazole.

**A residual, and its sign.** Kim also scanned at 45 h and 120 h, when the
metabolite share is higher than at 3.5 h. So Kim's parent-only constant may sit
slightly below the parent's own constant, which would make the chain
**over**-predict at 3.5 h. The observed errors are all negative, so this residual
does not explain them.

## Reading each result

**Aripiprazole: the PK layer contributes the bias.** P22.3 fed Ito's measured
plasma into the same EC50 and got MAE 0.0244 with bias −0.0048. Here the chain
predicts the plasma itself, and the errors become MAE 0.0406 and bias −0.0406,
all six negative. The cause is the plasma: it is predicted at 0.77–0.93 of
observed.

One plausible cause cannot be tested here. Ito's subjects were young Japanese
men, while Knights 2015's reference subject weighs 74.19 kg. A literature-bridged
record carries no weight rule (P25.2), so the chain cannot be moved off that
reference weight, and Ito prints no body weights. This is recorded as an open
attribution, not corrected.

**Quetiapine: the direct-effect form passes at peak and trough.** That is
notable given P21.2's doubt about it. The doubt still applies, but it is
contained. Nord's constant was fitted in the same once-daily peak-and-trough
design, so it absorbs the same hysteresis the test contains. This result does
not show that a direct-effect hyperbola is right for quetiapine at other scan
times or under twice-daily dosing.

Both errors are negative. A lag of occupancy behind plasma would raise trough
occupancy above the direct prediction, which is consistent with the trough
error. It would not explain the peak error.

Other limits:
- One independent group: the peak and trough scans are the same patients.
- Prediction is at the mean dose, 427 ± 69 mg.
- The values were read from the abstract. Its full text is closed.

**Olanzapine: the occupancy constant over-predicts, and the PK partly offsets
it.** Taking Mamo's measured 37.4 ng/mL straight through de Greef's hyperbola
gives 0.767, which is +0.076. The chain's lower plasma, 31.6 ng/mL at trough,
brings that down to +0.058.

Predicting at the mean dose should over-predict a group mean when doses vary
(15.2 ± 4.8 mg/day), because occupancy is concave in dose. The direction is
right, but the size of that effect is not computed, because the individual doses
are not printed.

Other limits:
- n = 1 group.
- Once-daily dosing is the ZYPREXA label schedule, not something Mamo printed.
- Mamo's scans were "proximal to trough" and are predicted at 24 h.

## How the observations were chosen

These rules were applied before any prediction was computed for the chosen
source:
- same tracer as the constant ([11C]raclopride);
- dose, schedule and scan time readable from the source;
- subjects disjoint from the constant's source.

Sources that failed the rules are listed in `unscored_candidates()` with the
reason.

**Disclosure.** Before choosing quetiapine's test, I computed the chain's
predictions for two schedule-ambiguous candidates while checking whether either
was usable:
- **Gefvert 1998**, 450 mg/day, 44% at 2 h: once daily predicts 0.563 and twice
  daily 0.435.
- **Kapur 2000**, 0–27% at 12–14 h over 150–600 mg/day: at 600 mg/day, once
  daily gives 0.305 and twice daily 0.206.

Neither abstract prints the schedule, and both papers are closed. One reading of
each would pass and the other would fail or be borderline. Choosing a schedule
would have meant choosing the result, so **both are left unscored and both
readings are printed** in the module.

The 0.10 threshold is P22.3's pre-registered value, reused unchanged.

## Failures, by missing layer

| drug | missing layer | why |
|---|---|---|
| clozapine | pk | Li 2012 co-fits norclozapine and prints no reference weight. Reeves 2024 needs a parallel-elimination slot and fails V13. The EC50 exists. |
| risperidone | active-metabolite | Every EC50 and every adult oral popPK is on the moiety or co-fits 9-OH-risperidone. |
| paliperidone | pk | Korell 2017 has zero-order release and no reference creatinine clearance. |
| ziprasidone | pk | No readable adult oral popPK in either corpus. |
| amisulpride | pk | No oral popPK record; it was outside P25.1's search. |

## Independence

- **Knights 2015** is the sponsor's model. Ito 2012 is academic (NIRS Chiba).
- **Zheng 2024's ka** is borrowed from Zhou 2015, which has AstraZeneca authors.
  Nord 2011's subjects came from an AstraZeneca database. Tauscher-Wisniewski
  2002 is an academic Toronto study.
- **Mamo 2008** was Lilly-sponsored (study F1D-EW-HGJW). The olanzapine PK
  comes from Alkermes' LYBALVI application. Both are industry data, from
  different sponsors and programmes.
