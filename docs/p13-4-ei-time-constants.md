# P13.4 — human E/I synaptic time constants

F-TRACT supplies separate excitatory and inhibitory synaptic-delay maps for
HCP-MMP1 parcels. `human.ei_time_constants.HumanEITimeConstants` preserves
finite values as measured human inputs and keeps missing values missing.
Callers must explicitly request `allow_fallback=True` to receive an
illustrative fallback; there is no silent imputation.

The release does not declare physical units for these maps. Values therefore
carry the source citation and an explicit `units` note, and must not be called
milliseconds without an independent source. Coverage is reported separately
for excitatory and inhibitory maps. This is a measured input adapter, not yet
a replacement of every `CircuitNode.tau` or E/I balance in the engine; that
wiring remains subject to the P13.5 stability review.
