# P18.2 — Held-out human-drug validation

The human layer now has explicit result records for held-out drug validation.
Results are classified as `scored`, `unavailable`, or `failed`; scored rows
must contain both predicted and observed values, while unavailable rows retain
the reason instead of being converted to a zero or omitted.

`summarize_heldout()` reports coverage and mean absolute error only over scored
rows. No held-out human observations are fabricated in this milestone. A
meaningful validation result still requires independent human drug data that
were not used for fitting.
