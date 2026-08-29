# P17.3 — Human dose-response fitting

The human layer now provides `fit_human_log_linear()`, a deliberately small
exploratory fitter for human observations. It requires at least three
positive, varied doses for one endpoint and one provenance source, then fits a
transparent response against log dose.

The fitter refuses insufficient data, mixed endpoints, mixed provenance, and
non-identifiable dose values. It does not use rodent observations, create
clinical parameters from priors, or imply that a simple log-linear curve is a
validated final pharmacodynamic model. Nonlinear fitting and held-out human
validation remain required before clinical use.
