# P17.1 — Human behavioral endpoints

The human layer now defines a small endpoint registry: cognitive performance,
motor activity, subjective sedation, and anxiety state. Each endpoint names a
human-observable measure and is explicitly distinct from rodent locomotion or
elevated-plus-maze outcomes.

`endpoint_score()` accepts node values and caller-supplied weights for a
transparent exploratory readout. It refuses empty weights because no human
endpoint magnitudes have been fitted yet. Therefore this milestone establishes
human endpoint contracts and prevents rodent endpoint substitution; it does
not claim validated dose-to-behavior predictions.
