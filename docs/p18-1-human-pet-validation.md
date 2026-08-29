# P18.1 — Human PET occupancy validation

The human layer now provides explicit PET occupancy observation and comparison
records. Each observation carries compound, target, dose, measured occupancy,
source, and optional uncertainty. Comparisons expose signed, absolute, and
standardized error without hiding missing uncertainty.

Predictions must be fractional occupancies in `[0, 1]`; invalid observations
and predictions are rejected. This establishes the validation contract but
does not fabricate PET observations or claim that any current compound has
passed validation. A real validation run still requires a cited human PET
dataset and an independently generated model prediction.
