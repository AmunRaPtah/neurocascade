# P14.3 — Human Kp,uu and species transfer

`human/kp_uu.py` defines the unbound brain-to-plasma coefficient separately
from total `Kp`. Diazepam has the measured human `Kp,uu,CSF = 0.79`; haloperidol
consumes no `Kp,uu` because its effective exposure is reserved for occupancy
inversion. Fentanyl has no defensible direct human value and is flagged as the
least-supported input. The three current human records are explicitly
`searched-and-empty`; no total Kp or rodent value is silently promoted to a
human receptor concentration.

`transfer_kp_uu()` permits a transferred value only when the caller supplies
both a non-human source value and an explicit positive transfer factor. Such a
value is labeled `inferred-species-transfer`, not measured human data. A direct
human value is labeled `measured-human`.

Species transfer is exposed as a fitted/scenario parameter rather than a
constant: the cited prior is approximately 1 for non-efflux substrates and
approximately 2–3 for P-gp substrates (Sato 2021). The CSF surrogate can
overpredict highly effluxed drugs; the reported human regression slope is
0.34. These limitations and the postmortem forensic route exclusion remain
part of the evidence boundary. Independent validation is still required
before clinical prediction.
