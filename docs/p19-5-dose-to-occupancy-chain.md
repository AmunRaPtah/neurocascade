# P19.5: the dose-to-occupancy chain for haloperidol

This is the first chain in the repository that runs end to end on human data.
`neurocascade.human.dose_to_occupancy.predict_occupancy` takes a dose in
milligrams, a route, a body weight and a time grid, and returns a predicted
striatal D2 occupancy-time curve together with the provenance of every layer
that produced it.

    dose (mg), route PO, body weight, t
        --Pilla Reddy 2013 two-compartment oral apparent set (P19.2)-->
    total plasma concentration C(t), nM
        --occupancy = C / (C + 1.8849 nM), fitted in P19.4 on the P19.3 data-->
    striatal D2 occupancy(t)
        --d2_effect, Emax 1, baseline 0-->
    D2Effect(t)

Nothing here has been compared to a measured occupancy. That comparison is
P19.6, and the chain was written before its result was known.

## Three design decisions, and why each is not the obvious one

### Two compartments, where the brief asked for three

The P19.5 brief specified `DosedThreeCompartmentPK`. There is no human
three-compartment haloperidol parameter set to put in it. P19.2 established
this: Holley 1983 fitted a three-compartment model and printed only clearance
and steady-state volume, Chang 1992 states that a three-compartment model
described its data best and never prints the micro-constants, and both are
paywalled, so no number was taken from either. The only defensible human set is
Pilla Reddy 2013's two-compartment oral one.

Building the three-compartment model the brief describes would have meant
inventing V3 and Q13. The chain instead reads the compartment count off the
sourced record and raises for anything other than two, naming what is missing.
The deviation is recorded here and in the module docstring rather than resolved
by manufacturing the parameters that would have satisfied the brief.

### No brain-partition step

The obvious chain multiplies plasma by a brain-to-plasma partition coefficient
and applies a receptor affinity to the resulting brain concentration. This chain
does not, and the reason is in the repository already: the haloperidol record in
`neurocascade.human.kp_uu` has `measurement_kind="absent"` and
`status="not-required"`, with the note that occupancy data should identify the
effective brain exposure.

That is the right call, because the P19.4 EC50 was fitted against *plasma*
concentrations measured in living people whose striatal occupancy was measured
at the same time. Brain penetration, plasma protein binding and competition with
endogenous dopamine are all already inside that constant. Multiplying by a Kp,uu
as well would apply the same correction twice.

The design is checked rather than assumed. `predict_occupancy` reads the Kp,uu
record on every call and raises if a value has since been added, because at that
point somebody has to decide which of the two layers carries brain penetration.
No `brain_partition` is attached to the PK model, so its
`receptor_concentration_factor` stays at 1.0 and it reports its receptor
compartment as "total plasma (no brain partition attached)".

### Apparent parameters, kept apparent

Pilla Reddy 2013 never identified bioavailability, so what it printed are
CL/F = 88 L/h, Q/F = 233 L/h, Vc/F = 669 L and Vp/F = 2500 L. None of the four
is identifiable on its own. The chain passes them with
`apparent_parameters=True`, which folds F into the values as bookkeeping and
forbids passing an F separately, rather than borrowing a bioavailability from
another study. A predicted concentration is therefore a prediction of
C/F-consistent exposure, and carries whatever error the true F carries. The `pk`
layer says so in its own text.

## Body weight

The entry point takes a body weight, and refuses to use it.

Pilla Reddy 2013 states an allometric exponent for clearance (0.75) and for
nothing else. Predicting at a weight other than the source's 70 kg reference
therefore needs a scaling rule for the volumes and for Q that the source does
not provide. Rather than quietly defaulting to the textbook exponents, the chain
raises unless the caller supplies a `WeightScaling(volume_exponent, q_exponent,
source)` and names where those exponents came from. Clearance is not covered by
the caller's rule: it keeps the exponent the source printed, through
`HumanPKRecord.CL_L_per_h_at_weight`.

This is the same discipline `HumanReceptorParameter.occupancy_constant_nM`
applies to an in-vitro correction factor: a number nobody measured may enter, but
only with a caller who has named it. When a rule is supplied, the
`weight-scaling` layer's provenance becomes `caller-supplied` and its citation
carries the caller's source string alongside the PK citation.

There is a specific trap this avoids. `DosedTwoCompartmentPK` takes per-kg
parameters, so the naive conversion is to divide the absolute values by the
requested weight and pass the weight through. Body weight then cancels out of the
kinetics entirely, which silently asserts an allometric exponent of 1 for
clearance where the source printed 0.75.

## Provenance

`OccupancyPrediction` carries a `layers` tuple, and the curve cannot be read off
it without also being handed the record each layer came from. The five layers
and what they currently claim:

| layer | provenance | source |
|---|---|---|
| `pk` | measured | Pilla Reddy 2013, PMID 24113674 |
| `weight-scaling` | structural, or caller-supplied | the PK source, plus whatever the caller names |
| `exposure` | structural | the `not-required` Kp,uu record |
| `occupancy` | fitted-in-repo | P19.4's fit on the P19.3 observations |
| `transduction` | structural | `neurocascade.human.d2_effect` |

`provenance` on the prediction is the weakest claim any layer makes. One
illustrative layer makes the whole prediction illustrative.

The P19.1 gate applies to the chain as a whole: `_gate_illustrative` refuses to
run if any layer's provenance is `illustrative` unless the caller passes
`allow_illustrative=True`, and the refusal carries the literal token
`ILLUSTRATIVE` that every other gate in the package uses. No layer of this chain
is illustrative today, so the gate refuses nothing at present. It exists so that
an illustrative constant added to any of these layers later cannot enter a
prediction unannounced. It is tested directly rather than through a synthetic
prediction, because there is no real illustrative layer to test it with.

`require_human_parameter("D2")` succeeds for real. It could not before P19.4,
when the D2 record had no in-vivo EC50 and was therefore not `usable`. Nothing in
the chain bypasses it.

## The transduction layer is not a behavioural claim

The brief asks for the `d2_effect` transducer, and `OccupancyPrediction.effects()`
applies it pointwise. With Emax 1 and baseline occupancy 0, neither of which has
been fitted in a human, `abs(effect)` equals occupancy exactly. It is the
occupancy curve on the transducer's sign convention (negative, because
haloperidol is an inverse agonist) and nothing more. A test asserts the identity,
so the claim that it is a re-expression rather than a prediction is checkable
rather than a promise in a docstring.

## What the chain refuses

Each of these is a test, not a comment:

- a PK record missing any of `V1_L`, `V2_L`, `Q12_L_per_h`, `CL_L_per_h` or
  `ka_per_min`, which is the acceptance criterion's "refuses to run when a
  required parameter is removed"
- a receptor record with the in-vivo EC50 removed, which is what a record holding
  only an in-vitro Ki looks like
- a receptor record whose EC50 is on a free-plasma basis, since the PK layer
  produces total plasma and no free-fraction step is applied
- route `IV`, because P19.2 recorded the haloperidol IV search as empty, and any
  route with no record at all
- a body weight off the reference weight with no stated scaling rule
- a Kp,uu record that has acquired a value
- a PK record for another drug or species, or a receptor record for another
  ligand
- a reporting grid that is not strictly increasing, starts before the dose, or
  has fewer than two points
- reading an occupancy at a time outside the window that was computed

## What it predicts

A single 5 mg oral dose in a 70 kg subject, on the default 72-hour grid:

    peak plasma 5.21 nM (1.96 ng/mL), at 195 min
    peak striatal D2 occupancy 0.734
    occupancy at 24 h 0.483

Across doses, single administration, peak occupancy:

| dose | Cmax (nM) | peak occupancy | occupancy at 24 h |
|---|---|---|---|
| 0.5 mg | 0.52 | 0.217 | 0.085 |
| 1 mg | 1.04 | 0.356 | 0.158 |
| 2 mg | 2.08 | 0.525 | 0.272 |
| 3 mg | 3.13 | 0.624 | 0.359 |
| 5 mg | 5.21 | 0.734 | 0.483 |
| 10 mg | 10.42 | 0.847 | 0.652 |
| 20 mg | 20.85 | 0.917 | 0.789 |

The saturation is the point of modelling occupancy rather than concentration:
forty times the dose is four times the peak occupancy.

These are single-dose predictions. The EC50 they use was fitted on depot
intramuscular steady-state observations, which is a mismatch carried forward from
P19.4 and stated in the `occupancy` layer's own text. Whether these numbers are
right is P19.6's question, and whether the 65 to 80 percent clinical window
appears in them is a question P19.6 is required to answer with the error it
actually gets.

## Integration

`solve_ivp` with LSODA at rtol 1e-10 and atol 1e-14, on whatever grid the caller
asks for. The grid is a display choice and not a model parameter: a test asserts
that a tenfold finer grid gives the same occupancy at the same times. The default
grid is 0 to 72 hours in 15-minute steps, which spans more than one predicted
terminal half-life of this parameter set (about 31 hours).

The solver can undershoot zero by a rounding-level amount on the flat tail. A
negative concentration is not a physical prediction and would make the hyperbola
non-monotonic, so the result is clipped at zero.
