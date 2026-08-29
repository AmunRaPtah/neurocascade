# P13.3 — macaque directed prior

`neurocascade.human.directed_prior` is the transfer boundary for macaque
FLNe/SLN tract-tracing data. Every `DirectedPriorEdge` is explicitly
`source_species="macaque"` and `provenance="inferred"`; it cannot be marked
measured. Overlaying a prior upgrades only absent human directions and never
overwrites F-TRACT measurements.

Sources:

- Markov et al. 2014, FLNe, *Cerebral Cortex*, PMID 23010748: 1,615 pathways,
  only 29 of 91 macaque areas injected.
- Markov et al. 2014, SLN hierarchy, *Journal of Comparative Neurology*,
  PMID 23983048.
- Mars et al. 2018, connectivity blueprints, *eLife*, PMID 29749930.

Reid et al. 2016 (PMID 26515902) found generally poor homology. Local
ipsilateral and homotopic connections are the best-supported cases; long-range
and heterotopic connections are weakest, especially in lateral/medial temporal
cortex, ACC, and TPJ. Human-expanded prefrontal and temporo-parietal cortex has
no macaque counterpart. These limitations must remain visible per edge or
region.

The requested Huang/Wang/Triebkorn/Jirsa bioRxiv 2026 preprint retry was
attempted, but the repository URL could not be verified in this build. Its
status is exposed as `JIRSA_RETRY_STATUS`; no validation claim is made.

No Markov data are fabricated or silently treated as human measurements. The
current implementation is the tested provenance-safe boundary awaiting a
verified blueprint mapping.
