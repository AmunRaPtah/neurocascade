# P16.3 — Human GABA-A operational agonism

`GABAAOperationalModel` maps receptor occupancy and intrinsic efficacy to a
bounded inhibitory-effect scale. It uses an explicit operational reserve,
baseline, and Emax; no efficacy value is inferred from a rodent assay.

The default `HUMAN_GABAA_OPERATIONAL_PRIOR` is an illustrative neutral prior,
not a validated benzodiazepine-specific human transducer. Human functional
occupancy/effect data are still required before this becomes a clinical
prediction parameter. The model is intentionally separate from PK, receptor
affinity, and circuit wiring.
