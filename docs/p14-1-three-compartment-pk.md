# P14.1 — three-compartment dosed PK

`DosedThreeCompartmentPK` adds a central compartment and two peripheral
compartments with direct `Q12` and `Q13` intercompartmental clearances. IV
exposure uses three concentration states; IP, SC, and PO exposure add a depot
state. Thus administered dose remains inside the ODE state vector consumed by
the existing engine.

The implementation is structural, not a claim that a parameter set is already
validated. Human fentanyl literature reports direct intercompartmental
clearances, so this model does not infer them from terminal half-life. Named
human parameter sets and validation remain P14.2 and P14.4.
