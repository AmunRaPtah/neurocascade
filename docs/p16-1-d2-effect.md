# P16.1 — D2 occupancy-limited effect

The human layer now exposes a small, bounded D2 effect transform:

`effect = -Emax * max(0, occupancy - baseline_occupancy)`

The model distinguishes receptor occupancy from effect magnitude. Baseline
occupancy produces no drug-induced effect, and the effect cannot exceed the
configured `Emax`. The direction is explicitly inhibitory by default.

This is a transducer contract, not a clinical calibration: human PET-derived
occupancy and an independently sourced D2 effect scale are still required for
an executable human prediction. No rodent affinity or behavioral magnitude is
introduced.
