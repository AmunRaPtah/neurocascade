# P17.4 — Human endpoint retirement

Human behavioral reporting now distinguishes retained human readouts from
rodent-specific assays.

Retained endpoints:

- Cognitive performance — reaction time and accuracy
- Motor activity — actigraphy or psychomotor tasks
- Subjective sedation — validated sedation scales
- Anxiety state — validated self-report and physiological scales

Retired endpoints:

- Elevated-plus-maze behavior
- Open-field locomotion

Retired endpoints cannot be requested through `require_human_endpoint()`.
They are not renamed or numerically translated into human measures. The
retained endpoints still require human-specific dose-response data before
they can support validated clinical predictions.
