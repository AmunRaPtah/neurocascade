"""Pharmacokinetics layer: what concentration the receptor layer actually sees.

Two families of model live here:

1. `OneCompartmentPK`, the v0 model, which starts from an *already-achieved*
   bolus concentration and only decays it. It is unchanged, both in behavior
   and in its `rhs(C)` signature.
2. `DosedOneCompartmentPK` / `DosedTwoCompartmentPK`, which start from an
   *administered dose* (mg/kg) and derive the concentration-time course from
   route, bioavailability `F`, absorption rate `ka`, volume of distribution
   and clearance. These are what let a literature statement like "haloperidol
   0.11 mg/kg PO" enter the model at all (see docs/architecture.md, "The
   dose-to-concentration bridge").

The ODE protocol every PK model in this module implements
-------------------------------------------------------

`engine.Simulation` integrates PK inside the one coupled state vector, so a
PK model is not allowed to precompute a concentration trace; it owns a slice
of the global state and reports that slice's derivative. The protocol is:

    model.n_states            -> int, how many state variables this model owns
    model.initial_state(dose) -> ndarray, shape (n_states,), state at t=0
    model.concentration_rhs(state) -> ndarray, shape (n_states,), d(state)/dt
    model.concentration(state)     -> the central-compartment (plasma/blood)
                                      concentration; by convention this is
                                      always `state[0]`, and it is
                                      `state[..., 0]`-safe so a whole
                                      (n_states, n_times) solution block can
                                      be reduced at once.
    model.receptor_concentration_factor -> float, the constant the central
                                      concentration is multiplied by to get
                                      the concentration the RECEPTOR layer
                                      binds. 1.0 when no brain partition is
                                      attached (plasma-driven, the pre-P1.5
                                      behavior); otherwise the partition's
                                      unbound-brain factor.
    model.receptor_concentration(state) -> that product.

Plasma is not what a receptor binds
-----------------------------------

The central compartment is plasma (or blood); receptors sit in brain tissue
behind the blood-brain barrier and bind only the *unbound* drug there. Every
model here therefore accepts an optional `brain_partition` (a
`brain_partition.BrainPartition`, whose numbers live in `drug_pk.py`), and
`engine.Simulation` drives occupancy from `receptor_concentration`, not from
`concentration`. With no partition attached the two are equal and every
pre-P1.5 result is bit-for-bit unchanged; `receptor_compartment` records
which case a given instance is in, so "plasma-driven" is a visible property
rather than a silent default. See docs/architecture.md, P1.5.

`dose` is whatever quantity that model's dosing is expressed in:
concentration (nM, or whatever unit the receptor constants use) for
`OneCompartmentPK`, administered dose in mg/kg for the dosed models. That is
why `Simulation.run` accepts a per-ligand *dose* rather than only a per-ligand
concentration: it is the model, not the engine, that knows how its dose
becomes state. Passing 0 means "no drug", identically in both families, so
every existing baseline-vs-dosed comparison keeps working.

Units
-----

Time is minutes everywhere in this codebase (`OneCompartmentPK.ke`, `koff`,
`kon` in the calibrated scenarios are all per-minute / nM^-1 min^-1), so
every rate constant here is per minute and clearance is expressed per minute.
Parameter names carry their units explicitly (`Vd_L_per_kg`,
`CL_mL_per_min_per_kg`, `ka_per_min`) because the literature reports these
quantities in at least four different unit conventions each, and a silently
mis-scaled clearance is exactly the kind of invented number this codebase
forbids. `per_hour_to_per_min` and friends below convert the common
literature forms; use them at the call site so the conversion is visible.

Concentration is reported in mg/L (== ug/mL) unless `molar_mass_g_per_mol`
is supplied, in which case the model converts to nM, the unit the calibrated
receptor constants use. The molar mass is a chemical fact, not a fitted
parameter, but it is still required to be passed explicitly rather than
guessed per drug, and `model.concentration_unit` records which unit a given
instance is producing.
"""

import math

import numpy as np

from .brain_partition import BrainPartition

LN2 = math.log(2.0)

#: Routes of administration this module models. IV is dosed directly into
#: the central compartment; the others are dosed into a depot compartment
#: absorbed first-order at `ka_per_min`.
ROUTES = ("IV", "IP", "SC", "PO")
EXTRAVASCULAR_ROUTES = ("IP", "SC", "PO")


# ---------------------------------------------------------------------------
# Unit conversion helpers. These exist so a literature value can be written at
# the call site in the units the paper actually reports, with the conversion
# visible in the diff, instead of a pre-converted number nobody can check.
# ---------------------------------------------------------------------------


def per_hour_to_per_min(k_per_hour):
    """Rate constant h^-1 -> min^-1."""
    return k_per_hour / 60.0


def half_life_to_rate_constant(t_half):
    """First-order half-life -> rate constant, in reciprocal of whatever time
    unit `t_half` is in (minutes, here, by convention)."""
    if t_half <= 0.0:
        raise ValueError("half-life must be positive")
    return LN2 / t_half


def clearance_L_per_h_per_kg_to_mL_per_min_per_kg(cl_L_per_h_per_kg):
    """Clearance L/h/kg (common clinical form) -> mL/min/kg (common
    preclinical form, and this module's parameter unit)."""
    return cl_L_per_h_per_kg * 1000.0 / 60.0


def concentration_mg_per_L_to_nM(c_mg_per_L, molar_mass_g_per_mol):
    """mg/L (== ug/mL) -> nM, at a given molar mass.

    mg/L = 1e-3 g/L; dividing by g/mol gives mol/L; 1 M = 1e9 nM, so the
    factor is 1e6 / molar_mass.
    """
    if molar_mass_g_per_mol <= 0.0:
        raise ValueError("molar mass must be positive")
    return c_mg_per_L * 1.0e6 / molar_mass_g_per_mol


def intercompartmental_clearance_from_terminal_rate(
    V1_L_per_kg, V2_L_per_kg, CL_mL_per_min_per_kg, beta_per_min
):
    """Solve for the inter-compartmental clearance Q (mL/min/kg) that makes a
    two-compartment model have a given terminal disposition rate constant
    `beta`, at a given V1, V2 and CL.

    Published rat/mouse PK papers routinely report V1 (or Vc), Vss, CL and a
    terminal half-life, and do NOT report Q, so a two-compartment model cannot
    be built from such a table without this step. Nothing is invented here:
    Q is the unique value consistent with the four numbers the paper already
    reported, obtained by inverting the standard identities

        alpha + beta = k10 + k12 + k21,   alpha * beta = k10 * k21

    (see `DosedTwoCompartmentPK.hybrid_rate_constants`). Substituting
    k10 = CL/V1, k12 = Q/V1, k21 = Q/V2 into beta's characteristic equation
    `beta^2 - (k10 + k12 + k21) beta + k10 k21 = 0` and solving the (linear
    in Q) result gives

        Q = beta (k10 - beta) / ( k10/V2 - beta (1/V1 + 1/V2) ).

    Caveat the caller is expected to record with the number: this inversion
    is ill-conditioned when beta is close to k10, because the denominator is
    then a small difference of comparable quantities. A terminal half-life
    reported to two significant figures can therefore pin Q only to within
    tens of percent, and the honest thing is to report that sensitivity
    rather than the single value (see docs/architecture.md, P1.3, for a
    worked example with haloperidol).

    Units: volumes L/kg, clearances mL/min/kg, rate constants min^-1.
    """
    if V1_L_per_kg <= 0.0 or V2_L_per_kg <= 0.0:
        raise ValueError("V1_L_per_kg and V2_L_per_kg must be positive")
    if CL_mL_per_min_per_kg <= 0.0:
        raise ValueError("CL_mL_per_min_per_kg must be positive")
    if beta_per_min <= 0.0:
        raise ValueError("beta_per_min must be positive")
    k10 = CL_mL_per_min_per_kg / (V1_L_per_kg * 1000.0)
    if beta_per_min >= k10:
        raise ValueError(
            "terminal rate constant beta (%g/min) must be smaller than "
            "k10 = CL/V1 (%g/min); a two-compartment model cannot have a "
            "terminal phase slower than its own elimination from the central "
            "compartment" % (beta_per_min, k10)
        )
    denominator = k10 / V2_L_per_kg - beta_per_min * (
        1.0 / V1_L_per_kg + 1.0 / V2_L_per_kg
    )
    if denominator <= 0.0:
        raise ValueError(
            "no positive inter-compartmental clearance reproduces "
            "beta=%g/min at V1=%g, V2=%g L/kg, CL=%g mL/min/kg; the reported "
            "parameter set is not consistent with a two-compartment model"
            % (beta_per_min, V1_L_per_kg, V2_L_per_kg, CL_mL_per_min_per_kg)
        )
    Q_L_per_min_per_kg = beta_per_min * (k10 - beta_per_min) / denominator
    return Q_L_per_min_per_kg * 1000.0


class _BrainPartitionMixin(object):
    """Shared plasma -> receptor-site concentration handling (P1.5).

    Every PK model in this module owns an optional `brain_partition`. Two
    invariants are enforced here rather than left to the caller:

    1. **The partition must be able to produce an unbound brain
       concentration.** A total-brain Kp with no measured brain unbound
       fraction is refused *at construction*, not at the first integration
       step, so the failure is a build-time error with the drug's own
       citation in the message rather than a silently wrong occupancy trace.
       (`BrainPartition` still exposes such a Kp for reporting total brain
       concentration; it just may not drive a receptor.)
    2. **Species must match.** A PK model that knows its species hands it to
       `BrainPartition.check_species`, so a rat partition cannot be attached
       to mouse kinetics. `OneCompartmentPK` has no species (its `ke` is
       whatever the caller sourced), so it passes None and the check is
       skipped - which is why the drug-specific builders in `drug_pk.py`,
       not this module, are where a species-matched pairing is guaranteed.
    """

    #: Set by `_attach_brain_partition`; None means "plasma-driven".
    brain_partition = None
    _receptor_factor = 1.0

    def _attach_brain_partition(self, brain_partition, species=None):
        if brain_partition is None:
            self.brain_partition = None
            self._receptor_factor = 1.0
            return
        if not isinstance(brain_partition, BrainPartition):
            raise TypeError(
                "brain_partition must be a BrainPartition (from "
                "neurocascade.brain_partition), got %r" % (brain_partition,)
            )
        brain_partition.check_species(species)
        # Raises UnusablePartitionError here, at construction, if the sourced
        # numbers cannot give an unbound brain concentration.
        self._receptor_factor = float(brain_partition.unbound_brain_factor)
        self.brain_partition = brain_partition

    @property
    def receptor_concentration_factor(self):
        return self._receptor_factor

    @property
    def receptor_compartment(self):
        """What `receptor_concentration` actually is, as a string, so that
        "this simulation drove occupancy with plasma" is inspectable instead
        of implicit."""
        if self.brain_partition is None:
            return "total plasma (no brain partition attached)"
        return "unbound brain"

    def receptor_concentration(self, state):
        """The concentration the receptor layer binds. Equal to
        `concentration(state)` when no partition is attached."""
        return self._receptor_factor * self.concentration(state)


class OneCompartmentPK(_BrainPartitionMixin):
    """First-order elimination from a single compartment: dC/dt = -ke * C.

    Dosing is expressed as an initial concentration (bolus) rather than an
    input term, since v0 only models single-bolus exposure.

    Retained unchanged as the model every existing calibrated scenario uses.
    `DosedOneCompartmentPK(route="IV", ...)` is exactly this equation with
    the initial concentration derived from an administered dose instead of
    stated directly (see `test_pk.py`'s reduction test).

    `species` (P11.2) is the provenance species of the KE the caller sourced,
    when there is one -- a human clinical half-life, say. It exists so that
    the species gate can read a claim off this model as off the dosed models;
    the default None means "no claim", which is the honest state of every
    illustrative ke and every structural use, and it is what every pre-P11.2
    caller gets. A partition attached here is now checked against this
    species exactly as `_DosedPKBase` checks its own.
    """

    n_states = 1
    concentration_unit = "unspecified"  # whatever unit the caller's C0 is in

    def __init__(self, ke, brain_partition=None, species=None):
        self.ke = ke
        self.species = None if species is None else str(species)
        self._attach_brain_partition(brain_partition, species=self.species)

    def rhs(self, C):
        return -self.ke * C

    # --- ODE protocol (see module docstring) -------------------------------

    def initial_state(self, dose):
        """`dose` here is the achieved bolus *concentration* at t=0."""
        return np.array([float(dose)], dtype=float)

    def concentration_rhs(self, state):
        return np.array([self.rhs(state[0])], dtype=float)

    def concentration(self, state):
        return state[0]

    @property
    def half_life_min(self):
        return LN2 / self.ke

    def __repr__(self):
        if self.brain_partition is None:
            return "OneCompartmentPK(ke=%g)" % self.ke
        return "OneCompartmentPK(ke=%g, brain_partition=%r)" % (
            self.ke,
            self.brain_partition,
        )


class _DosedPKBase(_BrainPartitionMixin):
    """Shared dose -> initial-condition arithmetic and route validation.

    The one modeling decision worth stating here rather than burying in each
    subclass: an extravascular dose is carried in a depot state expressed in
    *central-compartment concentration equivalents*, i.e. the depot state D
    is "the central concentration this undelivered drug would produce if it
    were absorbed instantaneously", so that

        dD/dt = -ka * D
        dC/dt = +ka * D + (disposition terms)

    is unit-consistent without threading a separate mass unit through the
    state vector. Bioavailability F is applied once, at the dose, so
    D(0) = F * dose / V_central; for a linear model this is algebraically
    identical to applying F to the absorption flux instead (F factors out),
    and it matches the usual NONMEM `F1` convention of scaling the dose.
    The (1 - F) fraction is simply never entered into the system: this model
    does not attempt to represent *where* it was lost (first pass, incomplete
    absorption, local degradation), only that it did not reach systemic
    circulation.

    `apparent_parameters=True` is the one sanctioned way to build an
    extravascular model without a measured F, and it is not a loophole: it
    is the statement that the volume and clearance being passed are the
    *apparent* ones, V/F and CL/F, as non-compartmental analysis of an
    extravascular study reports them. Those are exactly the quantities that
    determine the observed concentration-time course,

        C(t) = Dose / (V/F) * ka/(ka - ke) * (exp(-ke t) - exp(-ka t)),
        ke = (CL/F) / (V/F),

    with F appearing nowhere: for predicting what concentration a given
    mg/kg dose achieves, F is not separately identifiable and not separately
    needed. So `F` is set to 1 as *bookkeeping* (it has already been folded
    into the passed volume and clearance), the model records
    `apparent_parameters = True` and `F_is_measured = False`, and `repr`
    says so. What such a model cannot tell you is the true Vd, the true CL,
    or what fraction of the dose reached the circulation; asking it for
    those is asking for the one number that was divided out.

    Body weight is required, but note that it cancels exactly when dose,
    volume and clearance are all expressed per kg: C0 = F*dose_mg_kg*BW /
    (Vd_L_kg*BW). It is kept in the API because absolute amounts (mg dosed,
    L of distribution volume, mL/min of clearance) are what published PK
    tables report and what P1.6's Cmax/AUC validation will compare against,
    and because a future non-allometric parameterization would need it. It
    is never silently ignored: `dose_mg`, `V_central_L` and `CL_L_per_min`
    expose the absolute quantities it produces.
    """

    def __init__(
        self,
        route,
        body_weight_kg,
        F=None,
        ka_per_min=None,
        molar_mass_g_per_mol=None,
        name=None,
        apparent_parameters=False,
        brain_partition=None,
        species=None,
    ):
        route = str(route).upper()
        if route not in ROUTES:
            raise ValueError(
                "unknown route %r; expected one of %s" % (route, ", ".join(ROUTES))
            )
        if body_weight_kg is None or body_weight_kg <= 0.0:
            raise ValueError("body_weight_kg must be positive")
        apparent_parameters = bool(apparent_parameters)
        if apparent_parameters and route == "IV":
            raise ValueError(
                "apparent_parameters is meaningless for route 'IV': an "
                "intravenous dose has F = 1 by definition, so V/F is V and "
                "CL/F is CL"
            )

        if route == "IV":
            # Bioavailability of an intravenous bolus is 1 by definition, not
            # by assumption. Accepting an F != 1 here would let a caller
            # express "some of an IV dose never reached the circulation",
            # which is not a thing this model means.
            if F is not None and F != 1.0:
                raise ValueError("route 'IV' implies F = 1; got F=%r" % (F,))
            if ka_per_min is not None:
                raise ValueError(
                    "route 'IV' has no absorption phase; ka_per_min must be None"
                )
            F = 1.0
        elif apparent_parameters:
            # V/F and CL/F were passed, so F is already inside them. It must
            # not be passed again (that would apply it twice), and it is 1
            # only as bookkeeping.
            if F is not None and F != 1.0:
                raise ValueError(
                    "apparent_parameters=True means the volume and clearance "
                    "passed are already V/F and CL/F, so F must not be given "
                    "again; got F=%r" % (F,)
                )
            if ka_per_min is None or ka_per_min <= 0.0:
                raise ValueError(
                    "route %r requires a positive absorption rate ka_per_min" % route
                )
            F = 1.0
        else:
            # Deliberately NO default for F or ka on extravascular routes.
            # Both are measured, drug- and route-specific quantities;
            # defaulting F to 1.0 would silently invent the single number
            # that most changes the achieved concentration.
            if F is None:
                raise ValueError(
                    "route %r requires an explicit bioavailability F "
                    "(measured, drug- and route-specific; it is not "
                    "defaulted here on purpose), or apparent_parameters=True "
                    "if the volume and clearance passed are already V/F and "
                    "CL/F" % route
                )
            if not (0.0 < F <= 1.0):
                raise ValueError("F must be in (0, 1]; got %r" % (F,))
            if ka_per_min is None or ka_per_min <= 0.0:
                raise ValueError(
                    "route %r requires a positive absorption rate ka_per_min" % route
                )

        if molar_mass_g_per_mol is not None and molar_mass_g_per_mol <= 0.0:
            raise ValueError("molar_mass_g_per_mol must be positive when given")

        self.route = route
        self.body_weight_kg = float(body_weight_kg)
        self.F = float(F)
        self.ka_per_min = None if ka_per_min is None else float(ka_per_min)
        self.molar_mass_g_per_mol = molar_mass_g_per_mol
        self.name = name
        self.concentration_unit = "mg/L" if molar_mass_g_per_mol is None else "nM"
        #: True when the volume and clearance passed are V/F and CL/F rather
        #: than V and CL, so `F == 1.0` is bookkeeping and not a measurement.
        self.apparent_parameters = apparent_parameters
        #: False whenever F == 1 by convention rather than by measurement or
        #: by the definition of the intravenous route. Kept as its own flag
        #: because "F is 1" and "F was measured to be 1" are different
        #: claims, and only the second one licenses reporting a true Vd.
        self.F_is_measured = not apparent_parameters and route != "IV"
        #: Species this parameter set was measured in, when the builder knows
        #: it. Only used to refuse a cross-species brain partition; the
        #: kinetics themselves are already per-kg and species-agnostic in
        #: form (see "Body weight cancels", above).
        self.species = None if species is None else str(species)
        self._attach_brain_partition(brain_partition, species=self.species)

    @property
    def is_extravascular(self):
        return self.route in EXTRAVASCULAR_ROUTES

    def dose_mg(self, dose_mg_per_kg):
        """Absolute administered amount (mg) for a per-kg dose."""
        return dose_mg_per_kg * self.body_weight_kg

    def _to_concentration_unit(self, c_mg_per_L):
        if self.molar_mass_g_per_mol is None:
            return c_mg_per_L
        return concentration_mg_per_L_to_nM(c_mg_per_L, self.molar_mass_g_per_mol)

    def dose_equivalent_concentration(self, dose_mg_per_kg):
        """The central-compartment concentration this dose would produce if
        the bioavailable fraction were delivered instantaneously into the
        central compartment: F * dose / V_central, in `concentration_unit`.

        For an IV bolus this IS the initial concentration. For an
        extravascular route it is the depot state's initial value, i.e. the
        concentration that absorption will deliver over time (and never all
        at once, so Cmax is always strictly below it).
        """
        if dose_mg_per_kg < 0.0:
            raise ValueError("dose must be non-negative; got %r" % (dose_mg_per_kg,))
        c_mg_per_L = self.F * self.dose_mg(dose_mg_per_kg) / self.V_central_L
        return self._to_concentration_unit(c_mg_per_L)

    def auc_0_inf(self, dose_mg_per_kg):
        """Analytic AUC(0->inf) of the central compartment, in
        `concentration_unit` * min.

        For any linear compartmental model with first-order elimination only
        from the central compartment, AUC = F * Dose / CL exactly, regardless
        of route, absorption rate, or number of peripheral compartments (mass
        balance: all bioavailable drug eventually leaves via CL * C). Exposed
        for P1.6's validation against published AUC values.
        """
        if dose_mg_per_kg < 0.0:
            raise ValueError("dose must be non-negative; got %r" % (dose_mg_per_kg,))
        auc_mg_min_per_L = self.F * self.dose_mg(dose_mg_per_kg) / self.CL_L_per_min
        return self._to_concentration_unit(auc_mg_min_per_L)

    def concentration(self, state):
        """Central-compartment concentration; index 0 by convention for every
        model in this module, so an (n_states, n_times) block reduces too."""
        return state[0]


class DosedOneCompartmentPK(_DosedPKBase):
    """One-compartment model driven by an administered dose.

    IV bolus:

        C(0) = F * Dose / Vd          (F = 1)
        dC/dt = -ke * C,   ke = CL / Vd

    which is `OneCompartmentPK(ke=CL/Vd)` with the initial concentration
    derived rather than stated: the reduction is exact, not approximate, and
    `tests/test_pk.py::test_iv_bolus_reduces_exactly_to_one_compartment_pk`
    holds the whole coupled simulation to it.

    Extravascular (IP/SC/PO), first-order absorption from a depot:

        D(0) = F * Dose / Vd,  C(0) = 0
        dD/dt = -ka * D
        dC/dt = +ka * D - ke * C

    State layout: `[C]` for IV, `[C, D]` for an extravascular route (central
    concentration always first; see the module docstring's protocol).

    Parameters
    ----------
    route : one of ROUTES
    Vd_L_per_kg : apparent volume of distribution, L/kg
    CL_mL_per_min_per_kg : clearance, mL/min/kg
    body_weight_kg : body weight of the modeled animal, kg
    F : bioavailability, required for extravascular routes, forbidden (i.e.
        implicitly 1) for IV
    ka_per_min : first-order absorption rate constant, min^-1, required for
        extravascular routes, forbidden for IV
    molar_mass_g_per_mol : optional; when given, concentrations are produced
        in nM instead of mg/L
    apparent_parameters : set True when `Vd_L_per_kg` and
        `CL_mL_per_min_per_kg` are the *apparent* V/F and CL/F an
        extravascular non-compartmental analysis reports. F is then folded
        into them and must not be passed separately; see `_DosedPKBase`.
    """

    def __init__(
        self,
        route,
        Vd_L_per_kg,
        CL_mL_per_min_per_kg,
        body_weight_kg,
        F=None,
        ka_per_min=None,
        molar_mass_g_per_mol=None,
        name=None,
        apparent_parameters=False,
        brain_partition=None,
        species=None,
    ):
        super(DosedOneCompartmentPK, self).__init__(
            route=route,
            body_weight_kg=body_weight_kg,
            F=F,
            ka_per_min=ka_per_min,
            molar_mass_g_per_mol=molar_mass_g_per_mol,
            name=name,
            apparent_parameters=apparent_parameters,
            brain_partition=brain_partition,
            species=species,
        )
        if Vd_L_per_kg <= 0.0:
            raise ValueError("Vd_L_per_kg must be positive")
        if CL_mL_per_min_per_kg <= 0.0:
            raise ValueError("CL_mL_per_min_per_kg must be positive")
        self.Vd_L_per_kg = float(Vd_L_per_kg)
        self.CL_mL_per_min_per_kg = float(CL_mL_per_min_per_kg)

    # --- derived quantities ------------------------------------------------

    @property
    def V_central_L(self):
        return self.Vd_L_per_kg * self.body_weight_kg

    @property
    def CL_L_per_min(self):
        return self.CL_mL_per_min_per_kg * self.body_weight_kg / 1000.0

    @property
    def ke_per_min(self):
        """Elimination rate constant CL/Vd. Body weight cancels."""
        return self.CL_mL_per_min_per_kg / (self.Vd_L_per_kg * 1000.0)

    @property
    def terminal_half_life_min(self):
        return LN2 / self.ke_per_min

    def tmax_min(self):
        """Time of peak central concentration.

        IV bolus: 0 (the peak is the dose). Extravascular: the standard
        ln(ka/ke)/(ka - ke), the stationary point of the Bateman function;
        when ka == ke the limit is 1/ke.
        """
        if not self.is_extravascular:
            return 0.0
        ka, ke = self.ka_per_min, self.ke_per_min
        if ka == ke:
            return 1.0 / ke
        return math.log(ka / ke) / (ka - ke)

    # --- ODE protocol ------------------------------------------------------

    @property
    def n_states(self):
        return 2 if self.is_extravascular else 1

    def initial_state(self, dose_mg_per_kg):
        c0 = self.dose_equivalent_concentration(dose_mg_per_kg)
        if self.is_extravascular:
            return np.array([0.0, c0], dtype=float)
        return np.array([c0], dtype=float)

    def concentration_rhs(self, state):
        ke = self.ke_per_min
        if not self.is_extravascular:
            return np.array([-ke * state[0]], dtype=float)
        ka = self.ka_per_min
        C, D = state[0], state[1]
        return np.array([ka * D - ke * C, -ka * D], dtype=float)

    def __repr__(self):
        if self.apparent_parameters:
            return (
                "DosedOneCompartmentPK(route=%r, Vd/F=%g L/kg, "
                "CL/F=%g mL/min/kg, F folded in (apparent parameters))"
                % (self.route, self.Vd_L_per_kg, self.CL_mL_per_min_per_kg)
            )
        return "DosedOneCompartmentPK(route=%r, Vd=%g L/kg, CL=%g mL/min/kg, F=%g)" % (
            self.route,
            self.Vd_L_per_kg,
            self.CL_mL_per_min_per_kg,
            self.F,
        )


class DosedTwoCompartmentPK(_DosedPKBase):
    """Two-compartment (central + peripheral) model driven by a dose.

    In concentration form, with V1/V2 the central/peripheral volumes, CL the
    elimination clearance and Q the inter-compartmental clearance:

        dC1/dt = -(CL/V1) C1 - (Q/V1) C1 + (Q/V1) C2  [+ ka * D]
        dC2/dt = (Q/V2) (C1 - C2)
        dD/dt  = -ka * D                              [extravascular only]

    (the C2 return term carries Q/V1, not Q/V2, because the returning
    *amount* Q*C2 is diluted into V1; writing it as k21*C2 with k21 = Q/V2
    would be the amount-form equation, not the concentration-form one.)

    State layout: `[C1, C2]` for IV, `[C1, C2, D]` extravascular.

    Why two compartments matter here rather than being gratuitous detail: the
    calibrated mu-opioid scenario already had to choose between fentanyl's
    13 min redistribution half-life and its 219 min terminal half-life and
    could only represent one of them in a one-compartment collapse (see
    `scenarios.mu_opioid_pk_models_calibrated`). A two-compartment model
    represents both phases with their own literature-reported parameters
    instead of forcing that choice.
    """

    def __init__(
        self,
        route,
        V1_L_per_kg,
        V2_L_per_kg,
        Q_mL_per_min_per_kg,
        CL_mL_per_min_per_kg,
        body_weight_kg,
        F=None,
        ka_per_min=None,
        molar_mass_g_per_mol=None,
        name=None,
        apparent_parameters=False,
        brain_partition=None,
        species=None,
    ):
        super(DosedTwoCompartmentPK, self).__init__(
            route=route,
            body_weight_kg=body_weight_kg,
            F=F,
            ka_per_min=ka_per_min,
            molar_mass_g_per_mol=molar_mass_g_per_mol,
            name=name,
            apparent_parameters=apparent_parameters,
            brain_partition=brain_partition,
            species=species,
        )
        if V1_L_per_kg <= 0.0:
            raise ValueError("V1_L_per_kg must be positive")
        if V2_L_per_kg <= 0.0:
            raise ValueError("V2_L_per_kg must be positive")
        if Q_mL_per_min_per_kg < 0.0:
            raise ValueError("Q_mL_per_min_per_kg must be non-negative")
        if CL_mL_per_min_per_kg <= 0.0:
            raise ValueError("CL_mL_per_min_per_kg must be positive")
        self.V1_L_per_kg = float(V1_L_per_kg)
        self.V2_L_per_kg = float(V2_L_per_kg)
        self.Q_mL_per_min_per_kg = float(Q_mL_per_min_per_kg)
        self.CL_mL_per_min_per_kg = float(CL_mL_per_min_per_kg)

    # --- derived quantities ------------------------------------------------

    @property
    def V_central_L(self):
        return self.V1_L_per_kg * self.body_weight_kg

    @property
    def V_peripheral_L(self):
        return self.V2_L_per_kg * self.body_weight_kg

    @property
    def Vd_ss_L_per_kg(self):
        """Steady-state volume of distribution, V1 + V2."""
        return self.V1_L_per_kg + self.V2_L_per_kg

    @property
    def CL_L_per_min(self):
        return self.CL_mL_per_min_per_kg * self.body_weight_kg / 1000.0

    @property
    def k10_per_min(self):
        return self.CL_mL_per_min_per_kg / (self.V1_L_per_kg * 1000.0)

    @property
    def k12_per_min(self):
        return self.Q_mL_per_min_per_kg / (self.V1_L_per_kg * 1000.0)

    @property
    def k21_per_min(self):
        return self.Q_mL_per_min_per_kg / (self.V2_L_per_kg * 1000.0)

    def hybrid_rate_constants(self):
        """(alpha, beta), the distribution and terminal disposition rate
        constants, alpha >= beta > 0.

        These are the negated eigenvalues of the 2x2 disposition matrix, and
        satisfy the textbook identities

            alpha + beta = k10 + k12 + k21
            alpha * beta = k10 * k21

        Derived, not fitted: nothing here is a new number. With Q = 0 the
        peripheral compartment is disconnected and beta degenerates to 0
        (nothing ever returns), in which case the surviving positive root
        k10 is the elimination constant of the one-compartment model this
        collapses to.
        """
        b = self.k10_per_min + self.k12_per_min + self.k21_per_min
        c = self.k10_per_min * self.k21_per_min
        disc = max(b * b - 4.0 * c, 0.0)
        root = math.sqrt(disc)
        alpha = 0.5 * (b + root)
        beta = 0.5 * (b - root)
        if beta <= 0.0:
            # Q == 0: no peripheral exchange, so there is only one phase.
            return alpha, alpha
        return alpha, beta

    @property
    def terminal_half_life_min(self):
        return LN2 / self.hybrid_rate_constants()[1]

    @property
    def distribution_half_life_min(self):
        return LN2 / self.hybrid_rate_constants()[0]

    # --- ODE protocol ------------------------------------------------------

    @property
    def n_states(self):
        return 3 if self.is_extravascular else 2

    def initial_state(self, dose_mg_per_kg):
        c0 = self.dose_equivalent_concentration(dose_mg_per_kg)
        if self.is_extravascular:
            return np.array([0.0, 0.0, c0], dtype=float)
        return np.array([c0, 0.0], dtype=float)

    def concentration_rhs(self, state):
        k10 = self.k10_per_min
        k12 = self.k12_per_min
        k21 = self.k21_per_min
        C1, C2 = state[0], state[1]
        dC1 = -(k10 + k12) * C1 + k12 * C2
        dC2 = k21 * (C1 - C2)
        if not self.is_extravascular:
            return np.array([dC1, dC2], dtype=float)
        D = state[2]
        return np.array([dC1 + self.ka_per_min * D, dC2, -self.ka_per_min * D], dtype=float)

    def __repr__(self):
        return (
            "DosedTwoCompartmentPK(route=%r, V1=%g L/kg, V2=%g L/kg, "
            "Q=%g mL/min/kg, CL=%g mL/min/kg, F=%g)"
            % (
                self.route,
                self.V1_L_per_kg,
                self.V2_L_per_kg,
                self.Q_mL_per_min_per_kg,
                self.CL_mL_per_min_per_kg,
                self.F,
            )
        )


class _LegacySingleStatePK(_BrainPartitionMixin):
    """Adapter presenting a bare `rhs(C)`-only PK object (the pre-P1.1
    protocol) as an ODE-protocol model, so external/duck-typed PK models
    written against the v0 interface keep working inside `Simulation`.

    Such a model has no brain partition (it predates the concept), so it is
    plasma-driven and `receptor_concentration_factor` stays 1.0. Note that
    `__getattr__` below reads through to the wrapped object, so if a caller
    ever attaches a partition to their own model it would be visible but not
    honored; the factor here is deliberately the mixin's own, not the wrapped
    model's, because a bare `rhs(C)` object has no sanctioned way to declare
    one."""

    n_states = 1
    concentration_unit = "unspecified"

    def __init__(self, model):
        self.model = model
        self._attach_brain_partition(None)

    def initial_state(self, dose):
        return np.array([float(dose)], dtype=float)

    def concentration_rhs(self, state):
        return np.array([self.model.rhs(state[0])], dtype=float)

    def concentration(self, state):
        return state[0]

    def __getattr__(self, item):  # transparent read-through (e.g. .ke)
        return getattr(self.model, item)

    def __repr__(self):
        return "_LegacySingleStatePK(%r)" % (self.model,)


def as_ode_pk_model(model):
    """Return `model` if it implements the ODE protocol, else wrap it.

    `Simulation` calls this on every PK model it is given, so the engine
    only ever deals with one interface.
    """
    if hasattr(model, "concentration_rhs") and hasattr(model, "initial_state"):
        return model
    if hasattr(model, "rhs"):
        return _LegacySingleStatePK(model)
    raise TypeError(
        "PK model %r implements neither the ODE protocol "
        "(concentration_rhs/initial_state/n_states) nor the legacy rhs(C)" % (model,)
    )
