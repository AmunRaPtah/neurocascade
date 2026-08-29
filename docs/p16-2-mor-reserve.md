# P16.2 — Human μ-opioid receptor reserve

The human MOR effect layer now exposes `MORReserveModel`. It maps fractional
occupancy and intrinsic efficacy to a bounded response using an explicit
operational reserve factor:

`R = reserve × drive / (1 + (reserve − 1) × drive)`

where `drive = occupancy × efficacy`. The output is scaled between baseline
and Emax, so reserve can amplify submaximal occupancy without permitting an
unbounded or supra-Emax response.

`HUMAN_MOR_RESERVE_PRIOR` is deliberately labelled illustrative (`reserve =
1.5`). It is not a drug-specific human estimate and must be replaced or fitted
when matched human functional data are available. No rodent reserve value is
silently imported. This layer is also separate from PK, binding affinity, and
circuit gain; those remain independent parameters.
