# P15.1 — human receptor parameter boundary

The human layer now has an explicit receptor parameter registry for D2, MOR,
and GABA-A benzodiazepine sites. These records are human-scoped but remain
unusable until validated human occupancy/affinity values are transcribed.

The registry refuses missing targets and never falls back to rat or mouse
values. This is intentional: a human-first model must expose an evidence gap,
not turn a foreign-species constant into an apparently human prediction.

Human PET occupancy calibration is the next source of executable values and is
tracked in P16.1.
