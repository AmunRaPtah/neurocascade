# Prediction API

From a checkout, use `PYTHONPATH=src` with the repository's scientific
environment when running the snippets below.

NeuroCascade exposes two prediction levels. They should not be conflated:

1. `neurocascade.fingerprint.predict()` accepts a compound binding
   fingerprint and a dose/route/species input. It predicts the PK,
   receptor-driving brain concentration, occupancy and receptor activation
   time courses. It deliberately stops before circuit behavior because an
   arbitrary compound still needs a mechanism gain and a valid whole-brain
   receptor map.
2. `calibration.endpoint_forward` and
   `calibration.heldout_forward` operate on a modelled target or a composed
   set of target layers on the published connectome. They produce regional
   rates and named endpoint indices, but their illustrative mechanism gains
   are not automatically pharmacological predictions.

## Receptor-layer prediction from an arbitrary fingerprint

Every modelled target needs both a `Ki` and an efficacy. Efficacy is required
because a binding table does not say whether a ligand is an agonist,
antagonist, partial agonist, substrate or blocker. For a compound whose
drug-specific PK is not in `drug_pk.py`, supply a PK model and own its
provenance:

```python
from neurocascade import OneCompartmentPK
from neurocascade.fingerprint import BindingFingerprint, FingerprintEntry, predict

fingerprint = BindingFingerprint(
    "example_compound",
    entries={
        "D2": FingerprintEntry(
            target="D2",
            ki_nM=10.0,
            efficacy=0.0,
            efficacy_source="caller-supplied antagonist classification",
        )
    },
)

prediction = predict(
    fingerprint,
    dose=25.0,                         # achieved initial concentration, nM
    route="IV",
    species="rat",
    body_weight_kg=0.25,
    pk_model=OneCompartmentPK(ke=0.02), # caller owns this PK provenance
    n_times=241,
)

d2 = prediction["D2"]
print(d2.peak_occupancy, d2.time_of_peak_occupancy_min)
print(prediction.occupancy_ranking())
```

`d2.receptor_concentration` is the concentration actually supplied to the
site (unbound brain when the PK model carries a partition); `d2.occupancy` is
the compound's own occupancy; and `d2.activation` is the whole site's signed
activation, including competition. Transporters have no receptor occupancy;
their prediction exposes the extracellular transmitter trace instead.

The prediction object reports `brain_partition_status` and
`unmodelled`. A missing PK route, missing brain partition, missing efficacy or
unmodelled target is a reported limitation, never a silently substituted
number. `predict()` does not produce a behavioral endpoint ratio.

## Whole-brain endpoint analysis

For one target and one mechanism, build the P7.1 whole-brain model and read a
steady state or response curve:

```python
from neurocascade.calibration.endpoint_forward import build_whole_brain_model

model, provenance = build_whole_brain_model(
    "Gabra1", "connection_modulation", "gaba_a_whole_brain",
    resolution="regions",
)
baseline = model.baseline_rates
active_rates = model.rates(0.5)  # x = mechanism_gain * activation
anxiety_ratio = model.index("anxiety", active_rates) / model.index(
    "anxiety", baseline
)
curve = model.response_curve(
    ["locomotion", "anxiety"],
    x_grid=[0.0, 0.5, 1.0, 2.0, 5.0],
)
print(provenance["n_nodes"], anxiety_ratio, curve("anxiety", 1.0))
```

The `x` argument is a mechanism product, not a dose. To connect it to a
compound, the caller must supply a justified activation and a mechanism gain.
The whole-brain GABA-A gain used in P7.3's reachability plots is bounded by an
illustrative prior and was not accepted as a validated behavioral parameter.

For multi-target held-out composition, use the shared-network model:

```python
from neurocascade.calibration.heldout_forward import build_multi_target_model

model, provenance = build_multi_target_model(
    ["D2", "GABA_A_BZD", "MOR"],
    "heldout_example",
    resolution="regions",
)
rates = model.rates({"D2": 0.1, "GABA_A_BZD": 0.05})
indices = model.indices({"D2": 0.1, "GABA_A_BZD": 0.05})
```

This composes target-layer `x` values on one connectome. It is useful for
structural reachability and sensitivity analysis; it is not a fitted
compound-to-behavior prediction unless every upstream PK, partition,
activation and gain is available and validated.

## What the API does not claim

- It does not infer efficacy from `Ki`.
- It does not convert an arbitrary administered dose to brain exposure without
  a species/route-specific PK and brain-partition record.
- It does not turn transcript expression into an absolute receptor density;
  the relative-density conversion is an explicit assumption with measured
  rank/scale error and terminal-field mismatch flags.
- It does not fit the illustrative node parameters, connectome scale,
  endpoint weight magnitudes or most mechanism gains.
- It does not score the seven P8.2 held-out compounds end to end: the
  committed result is that their observed routes lack the required complete
  chain. See [`heldout-failure-analysis.md`](heldout-failure-analysis.md).

The complete calibration inventory is in the README's parameter-status table;
the machine-readable target and `kon` standings are available through
`fingerprint.target_report()`.
