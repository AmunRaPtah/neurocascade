"""Pharmacokinetics: the dose -> plasma-concentration layer.

Every model in this module is ODE-native. A model never precomputes a
concentration-time trace and hands it to the rest of the stack; it exposes
a right-hand side that `engine.Simulation` evaluates inside the single
coupled state vector, so concentration, receptor occupancy and circuit
state all advance together (see "Why one coupled ODE system instead of
separate stages" in docs/architecture.md).

Units, matching the rest of the codebase (docs/architecture.md, "Calibrated"
sections of scenarios.py): concentration in nM, amounts in nmol, time in
minutes. Dose is mg/kg, body weight kg, volumes L/kg, clearances
mL/min/kg, absorption rate min^-1. Those are the units the rodent PK
literature actually reports, so a published parameter set can be typed in
without a conversion step that could silently go wrong; the conversion to
the model's internal nM/nmol/min happens once, here, in `_dose_nmol` and
in each model's derived rate constants.

This module contains NO drug-specific numbers. Every parameter is supplied
by the caller; literature parameter sets for specific drugs, routes and
species belong in `scenarios.py` alongside the binding constants, with
their citations (roadmap P1.2-P1.4).

The PK model protocol `engine.Simulation` relies on:

    n_aux                       int, how many auxiliary states this model
                                owns beyond its plasma concentration
                                (depot amount, peripheral amount, ...).
    concentration_rhs(C, aux)   -> (dC/dt, d(aux)/dt), the model's
                                contribution to the coupled RHS.
    initial_state(bolus_concentration)
                                -> (C0, aux0) at t = 0.

`OneCompartmentPK` implements all three, so the two APIs are one API;
`Simulation` also still accepts a bare object with only the original
`rhs(C)` method, which is what `OneCompartmentPK` exposed before the
dose-driven models existed.
"""

import numpy as np


#: Routes this module models. "iv" places the bioavailable dose directly in
#: the central compartment; every other route places it in a depot
#: compartment that empties into the central one at first-order rate `ka`.
ROUTES = ("iv", "ip", "sc", "po")

_EMPTY = np.zeros(0)


def _dose_nmol(dose_mg_per_kg, body_weight_kg, molar_mass_g_per_mol):
    """Administered dose (mg/kg) -> amount in nmol.

    mg / (g/mol) = mmol (a g/mol is a mg/mmol), and 1 mmol = 1e6 nmol.
    """
    dose_mg = dose_mg_per_kg * body_weight_kg
    return (dose_mg / molar_mass_g_per_mol) * 1.0e6


def _validate_route(route, bioavailability, ka_per_min):
    if route not in ROUTES:
        raise ValueError("route must be one of %s, got %r" % (list(ROUTES), route))
    if route == "iv":
        # Bioavailability is 1 by definition for an intravascular dose: the
        # whole dose reaches the systemic circulation because that is where
        # it was put. Accepting F != 1 here would let a caller quietly
        # invent a loss term with no mechanism behind it.
        if bioavailability != 1.0:
            raise ValueError(
                "route='iv' requires bioavailability=1.0 (IV bioavailability "
                "is 1 by definition); got %r" % (bioavailability,)
            )
        if ka_per_min is not None:
            raise ValueError(
                "route='iv' has no absorption phase, so ka_per_min is "
                "meaningless; leave it None"
            )
    else:
        if ka_per_min is None:
            raise ValueError(
                "route=%r is extravascular and needs a first-order "
                "absorption rate ka_per_min (min^-1)" % (route,)
            )
        if ka_per_min <= 0.0:
            raise ValueError("ka_per_min must be positive, got %r" % (ka_per_min,))
    if not (0.0 <= bioavailability <= 1.0):
        raise ValueError(
            "bioavailability must lie in [0, 1], got %r" % (bioavailability,)
        )


def _reject_bolus(model_name, bolus_concentration):
    """A dose-driven model computes its own t=0 state from the dose it was
    given. Handing it a nonzero bolus concentration as well would mean two
    unrelated dosing statements for one ligand, so it is an error rather
    than a silent override in either direction.
    """
    if bolus_concentration:
        raise ValueError(
            "%s already carries its own administered dose; passing a nonzero "
            "bolus concentration (%r) for the same ligand would dose it twice. "
            "Pass 0.0 (or omit dose_concentrations entirely)."
            % (model_name, bolus_concentration)
        )


class OneCompartmentPK:
    """First-order elimination from a single compartment: dC/dt = -ke * C.

    Dosing is expressed as an initial concentration (bolus) rather than an
    input term: this is the original v0 model, kept unchanged because every
    calibrated scenario and every published *occupancy* validation target in
    this codebase is anchored to an achieved plasma concentration, not to an
    administered dose. `OneCompartmentDosePK(route="iv", ...)` is this same
    equation reached from a dose instead (see docs/architecture.md,
    "Phase 1: the dose -> concentration extension").
    """

    n_aux = 0

    def __init__(self, ke):
        self.ke = ke

    def rhs(self, C):
        return -self.ke * C

    def concentration_rhs(self, C, aux=_EMPTY):
        return -self.ke * C, _EMPTY

    def initial_state(self, bolus_concentration=0.0):
        return bolus_concentration, _EMPTY


class _DosePKBase(object):
    """Shared dose bookkeeping for the administered-dose models.

    Holds the route, the bioavailable amount in nmol, and the derived rate
    constants, so the one- and two-compartment models below differ only in
    their distribution structure.
    """

    def __init__(
        self,
        dose_mg_per_kg,
        body_weight_kg,
        molar_mass_g_per_mol,
        route,
        bioavailability=1.0,
        ka_per_min=None,
    ):
        if dose_mg_per_kg < 0.0:
            raise ValueError("dose_mg_per_kg must be non-negative")
        if body_weight_kg <= 0.0:
            raise ValueError("body_weight_kg must be positive")
        if molar_mass_g_per_mol <= 0.0:
            raise ValueError("molar_mass_g_per_mol must be positive")
        _validate_route(route, bioavailability, ka_per_min)

        self.dose_mg_per_kg = dose_mg_per_kg
        self.body_weight_kg = body_weight_kg
        self.molar_mass_g_per_mol = molar_mass_g_per_mol
        self.route = route
        self.bioavailability = bioavailability
        self.ka = ka_per_min

        #: nmol of drug that reaches the systemic circulation. F multiplies
        #: the *amount* entering the body, the standard compartmental
        #: convention: an extravascular dose loses the non-bioavailable
        #: fraction before it ever reaches plasma, so it never enters the
        #: state vector at all, rather than being eliminated from it.
        self.bioavailable_nmol = self.bioavailability * _dose_nmol(
            dose_mg_per_kg, body_weight_kg, molar_mass_g_per_mol
        )
        #: True when the dose starts in a depot compartment that empties
        #: into plasma at rate ka. IV has no depot state at all rather than
        #: a permanently-empty one, which is what makes the IV case reduce
        #: *exactly*, state-vector and all, to `OneCompartmentPK`.
        self.has_depot = route != "iv"

    def _central_volume_L(self, volume_L_per_kg):
        return volume_L_per_kg * self.body_weight_kg

    def _clearance_L_per_min(self, clearance_mL_per_min_per_kg):
        return clearance_mL_per_min_per_kg * self.body_weight_kg / 1000.0


class OneCompartmentDosePK(_DosePKBase):
    """One-compartment model driven by an administered dose.

    IV::

        dC/dt = -ke * C,                      C(0) = F * dose_nmol / V

    Extravascular (ip / sc / po), with a depot amount `A_d` in nmol::

        dA_d/dt = -ka * A_d,                  A_d(0) = F * dose_nmol
        dC/dt   = ka * A_d / V - ke * C,      C(0)   = 0

    where `ke = CL / V` (L/min over L, i.e. min^-1) and `V = Vd * weight`.

    Route is not itself a separate mechanism here: ip, sc and po share the
    identical first-order-absorption structure, and differ only through the
    values of `F` and `ka`, which is exactly how the literature reports the
    difference (an IP dose's first-pass loss shows up as a lower measured
    F, not as a different equation). The route label is carried so a
    parameter set can state which route it was measured on, and so IV can
    be structurally distinguished, since IV genuinely has no absorption
    phase.

    Parameters
    ----------
    dose_mg_per_kg, body_weight_kg : the administered dose and the animal.
    molar_mass_g_per_mol : needed to reach the codebase's nM state variable
        from a mass dose; a chemical property of the compound.
    volume_of_distribution_L_per_kg : Vd.
    clearance_mL_per_min_per_kg : CL.
    route : one of ROUTES.
    bioavailability : F, must be 1.0 for IV.
    ka_per_min : first-order absorption rate; required unless route is IV,
        forbidden when it is.
    """

    def __init__(
        self,
        dose_mg_per_kg,
        body_weight_kg,
        molar_mass_g_per_mol,
        volume_of_distribution_L_per_kg,
        clearance_mL_per_min_per_kg,
        route="iv",
        bioavailability=1.0,
        ka_per_min=None,
    ):
        _DosePKBase.__init__(
            self,
            dose_mg_per_kg=dose_mg_per_kg,
            body_weight_kg=body_weight_kg,
            molar_mass_g_per_mol=molar_mass_g_per_mol,
            route=route,
            bioavailability=bioavailability,
            ka_per_min=ka_per_min,
        )
        if volume_of_distribution_L_per_kg <= 0.0:
            raise ValueError("volume_of_distribution_L_per_kg must be positive")
        if clearance_mL_per_min_per_kg < 0.0:
            raise ValueError("clearance_mL_per_min_per_kg must be non-negative")

        self.volume_of_distribution_L_per_kg = volume_of_distribution_L_per_kg
        self.clearance_mL_per_min_per_kg = clearance_mL_per_min_per_kg
        self.V = self._central_volume_L(volume_of_distribution_L_per_kg)
        self.CL = self._clearance_L_per_min(clearance_mL_per_min_per_kg)
        #: elimination rate constant, min^-1: the single number
        #: `OneCompartmentPK` takes directly.
        self.ke = self.CL / self.V

        self.n_aux = 1 if self.has_depot else 0

    def concentration_rhs(self, C, aux=_EMPTY):
        if not self.has_depot:
            # Structurally identical to OneCompartmentPK.concentration_rhs.
            return -self.ke * C, _EMPTY
        depot = aux[0]
        d_depot = -self.ka * depot
        dC = self.ka * depot / self.V - self.ke * C
        return dC, np.array([d_depot])

    def initial_state(self, bolus_concentration=0.0):
        _reject_bolus("OneCompartmentDosePK", bolus_concentration)
        if not self.has_depot:
            return self.bioavailable_nmol / self.V, _EMPTY
        return 0.0, np.array([self.bioavailable_nmol])

    def __repr__(self):
        return "OneCompartmentDosePK(route=%r, dose=%g mg/kg, ke=%g/min)" % (
            self.route,
            self.dose_mg_per_kg,
            self.ke,
        )


class TwoCompartmentDosePK(_DosePKBase):
    """Two-compartment (central + peripheral) model driven by a dose.

    With central concentration `C` (nM), peripheral amount `A_p` (nmol),
    optional depot amount `A_d` (nmol), central volume `V1 = V1/kg * weight`,
    peripheral volume `V2`, elimination clearance `CL` and
    intercompartmental clearance `Q` (both L/min)::

        dA_d/dt = -ka * A_d                                  (extravascular only)
        dC/dt   = ka * A_d / V1 - (CL / V1) * C - (Q / V1) * (C - A_p / V2)
        dA_p/dt = Q * (C - A_p / V2)

    This is the model that matters for the drugs in this codebase: fentanyl,
    haloperidol and diazepam all show a distribution phase distinct from
    their terminal elimination phase, and a one-compartment fit forced onto
    a two-phase curve misstates early concentrations, which are exactly the
    concentrations receptor occupancy is most sensitive to.

    Setting `Q = 0` removes the peripheral compartment's influence
    entirely, so this class reduces to `OneCompartmentDosePK` on the same
    parameters (tested).

    Auxiliary state order is `[depot, peripheral]` extravascularly and
    `[peripheral]` for IV; there is no dead depot state for IV.
    """

    def __init__(
        self,
        dose_mg_per_kg,
        body_weight_kg,
        molar_mass_g_per_mol,
        central_volume_L_per_kg,
        peripheral_volume_L_per_kg,
        clearance_mL_per_min_per_kg,
        intercompartmental_clearance_mL_per_min_per_kg,
        route="iv",
        bioavailability=1.0,
        ka_per_min=None,
    ):
        _DosePKBase.__init__(
            self,
            dose_mg_per_kg=dose_mg_per_kg,
            body_weight_kg=body_weight_kg,
            molar_mass_g_per_mol=molar_mass_g_per_mol,
            route=route,
            bioavailability=bioavailability,
            ka_per_min=ka_per_min,
        )
        if central_volume_L_per_kg <= 0.0:
            raise ValueError("central_volume_L_per_kg must be positive")
        if peripheral_volume_L_per_kg <= 0.0:
            raise ValueError("peripheral_volume_L_per_kg must be positive")
        if clearance_mL_per_min_per_kg < 0.0:
            raise ValueError("clearance_mL_per_min_per_kg must be non-negative")
        if intercompartmental_clearance_mL_per_min_per_kg < 0.0:
            raise ValueError(
                "intercompartmental_clearance_mL_per_min_per_kg must be "
                "non-negative"
            )

        self.central_volume_L_per_kg = central_volume_L_per_kg
        self.peripheral_volume_L_per_kg = peripheral_volume_L_per_kg
        self.clearance_mL_per_min_per_kg = clearance_mL_per_min_per_kg
        self.intercompartmental_clearance_mL_per_min_per_kg = (
            intercompartmental_clearance_mL_per_min_per_kg
        )

        self.V1 = self._central_volume_L(central_volume_L_per_kg)
        self.V2 = self._central_volume_L(peripheral_volume_L_per_kg)
        self.CL = self._clearance_L_per_min(clearance_mL_per_min_per_kg)
        self.Q = self._clearance_L_per_min(
            intercompartmental_clearance_mL_per_min_per_kg
        )
        self.ke = self.CL / self.V1

        self._depot_index = 0 if self.has_depot else None
        self._peripheral_index = 1 if self.has_depot else 0
        self.n_aux = 2 if self.has_depot else 1

    def concentration_rhs(self, C, aux):
        peripheral = aux[self._peripheral_index]
        C_peripheral = peripheral / self.V2
        distribution_flux = self.Q * (C - C_peripheral)  # nmol/min, central -> peripheral

        dC = -self.ke * C - distribution_flux / self.V1
        d_aux = np.empty(self.n_aux)
        d_aux[self._peripheral_index] = distribution_flux
        if self.has_depot:
            depot = aux[self._depot_index]
            d_aux[self._depot_index] = -self.ka * depot
            dC += self.ka * depot / self.V1
        return dC, d_aux

    def initial_state(self, bolus_concentration=0.0):
        _reject_bolus("TwoCompartmentDosePK", bolus_concentration)
        aux0 = np.zeros(self.n_aux)
        if self.has_depot:
            aux0[self._depot_index] = self.bioavailable_nmol
            return 0.0, aux0
        return self.bioavailable_nmol / self.V1, aux0

    def __repr__(self):
        return "TwoCompartmentDosePK(route=%r, dose=%g mg/kg, ke=%g/min, Q=%g L/min)" % (
            self.route,
            self.dose_mg_per_kg,
            self.ke,
            self.Q,
        )
