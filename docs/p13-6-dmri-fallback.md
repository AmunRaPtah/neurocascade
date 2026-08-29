# P13.6 — dMRI fallback demotion

The ENIGMA/HCP diffusion-MRI matrix remains available through
`human.dmri_fallback.DMRIFallback`, but it is explicitly a fallback. Every
nonzero directed interpretation is `inferred`; the fallback can satisfy an
`INFERRED` requirement but always refuses a `MEASURED` requirement.

The quantitative tractography limitation is recorded from Maier-Hein et al.
2017 (PMID 29116093): 36 +/- 17% of reconstructed streamlines connected
regions that were not connected, with 88 +/- 58 invalid bundles per tractogram
and bundle precision of 23 +/- 9%.

The current F-TRACT dataset uses HCP-MMP1 (360 parcels), while ENIGMA uses
Desikan-Killiany plus subcortex (82 regions). Since no verified crosswalk is
present, comparison returns `not_comparable` rather than inventing one. This
is an honest intermediate result; a future crosswalk must preserve measured
F-TRACT evidence and leave dMRI edges flagged inferred.
