"""Dose to striatal D2 occupancy for human haloperidol (P19.5).

The first end-to-end human chain in this repository. One entry point,
:func:`predict_occupancy`, maps

    dose (mg), route, body weight (kg), time (min)
        --HumanPKRecord for haloperidol PO, transcribed in P19.2-->
    total plasma concentration C(t), in nM
        --occupancy = Emax * C / (C + EC50), fitted in P19.4, and since P21.1
          read out of the drug-keyed registry in `human.d2_ec50` together with
          the Emax convention, plasma basis and analyte it was fitted under.
          Haloperidol's row pins Emax at 1, so the curve is the same hyperbola
          it always was-->
    striatal D2 occupancy(t)
        --d2_effect, a unit-Emax re-expression-->
    D2Effect(t)

Three things about that chain are deliberate and are the reason this module
exists rather than a two-line helper.

**There is no brain-partition step.** The obvious way to write this would be to
multiply plasma by a Kp,uu to obtain a brain concentration and then apply a
receptor constant. This chain does not, because there is no measured human
haloperidol Kp,uu: the record in :mod:`neurocascade.human.kp_uu` is marked
``not-required`` with the note that occupancy data should identify the effective
brain exposure. The P19.4 EC50 was fitted against *plasma* concentrations in
living people, so brain penetration, plasma protein binding and competition with
endogenous dopamine are already inside it. Inserting a partition step would apply
those corrections twice. :func:`predict_occupancy` checks that the Kp,uu record
is still empty and refuses to run if a value has since been added, because at
that point the double-counting question has to be answered rather than assumed.

**Two compartments, not three.** The P19.5 brief asks for
``DosedThreeCompartmentPK``. P19.2 established that no human three-compartment
haloperidol parameter set is available: Holley 1983 fitted one and prints only CL
and Vss, Chang 1992 states a three-compartment model won and never prints its
micro-parameters, and both are paywalled. The only defensible human set is Pilla
Reddy 2013's two-compartment oral one. Building a three-compartment model here
would mean inventing V3 and Q13, so this module dispatches on the compartment
count the sourced record actually declares and raises for any other value. The
deviation from the brief is the finding, not a shortcut around it.

**The parameters are apparent (X/F).** Pilla Reddy 2013 never identified F, so
CL/F, Q/F, Vc/F and Vp/F are each unidentifiable individually. The chain passes
them to ``DosedTwoCompartmentPK`` with ``apparent_parameters=True``, which folds F
in as bookkeeping rather than letting a bioavailability borrowed from another
study enter unannounced. A predicted concentration from this chain is therefore a
prediction of C/F-consistent exposure, and its absolute scale carries whatever
error the true F carries.

Nothing here has been compared to a measured occupancy. That is P19.6.
"""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from ..pk import DosedTwoCompartmentPK
from .d2_effect import d2_effect_for_drug
from .active_metabolites import active_metabolites_for, require_parent_only_path
from .kp_uu import kp_uu_records
from .pet_validation import HALOPERIDOL_MW_G_PER_MOL
from .pk_parameters import human_pk_records
from .receptor_parameters import require_human_parameter

#: What a layer's numbers rest on. ``illustrative`` is the P19.1 category: a
#: constant nobody measured, which may only be consumed through an explicit
#: opt-in. The other four are all claims about where a real number came from.
PROVENANCE_LEVELS = (
    "measured",
    "fitted-in-repo",
    "structural",
    "caller-supplied",
    "illustrative",
)

_DRUG = "haloperidol"
_TARGET = "D2"


@dataclass(frozen=True)
class WeightScaling:
    """A caller-supplied rule for moving volumes and inter-compartmental flow
    off the source's reference weight.

    Pilla Reddy 2013 states an allometric exponent for clearance and for nothing
    else, so a prediction at a weight other than the reference weight needs a
    rule for the volumes and for Q that the source does not provide. Rather than
    default to the textbook exponents, this chain makes the caller state them and
    name where they came from, the same discipline
    :meth:`HumanReceptorParameter.occupancy_constant_nM` applies to an in-vitro
    correction factor. Clearance is *not* covered by this rule: it keeps the
    exponent the source itself printed.
    """

    volume_exponent: float
    q_exponent: float
    source: str

    def __post_init__(self):
        if not self.source:
            raise ValueError(
                "source must name where the volume and Q exponents came from"
            )
        for name in ("volume_exponent", "q_exponent"):
            value = getattr(self, name)
            if not np.isfinite(value):
                raise ValueError("%s must be finite" % name)


@dataclass(frozen=True)
class ChainLayer:
    """One layer of the chain and where its numbers came from."""

    name: str
    status: str
    detail: str
    citation: str
    provenance: str

    def __post_init__(self):
        if self.provenance not in PROVENANCE_LEVELS:
            raise ValueError(
                "provenance %r for layer %r is not one of %s"
                % (self.provenance, self.name, ", ".join(PROVENANCE_LEVELS))
            )


@dataclass(frozen=True)
class OccupancyPrediction:
    """A predicted occupancy-time curve and the provenance of every layer used.

    Deliberately carries `layers`: the numbers cannot be read off this object
    without also being handed the record each one came from, for the same reason
    :class:`neurocascade.legacy.pk_validation.ValidationResult` refuses to hand
    back a comparison without its tier and source.
    """

    drug: str
    target: str
    dose_mg: float
    route: str
    body_weight_kg: float
    times_min: np.ndarray
    plasma_nM: np.ndarray
    occupancy: np.ndarray
    concentration_basis: str
    receptor_compartment: str
    ec50_nM: float
    layers: tuple
    #: :class:`ActiveMetabolite` records for this drug whose disposition is
    #: ``bounded``: metabolites that occupy the same receptor and were cleared
    #: by an arithmetic bound rather than by being absent. A ``refused`` record
    #: never reaches this field, because the prediction does not get built.
    #:
    #: These are not a layer. A layer's provenance is a claim about a number the
    #: prediction consumed, and the chain consumes nothing from these records:
    #: the parent-only EC50 already absorbs whatever the metabolite contributed
    #: in the regimen it was fitted in. What the records carry is the size of
    #: the residual risk if the driven regimen's exposure ratio differs from the
    #: fitted one, which belongs in the caveats rather than in the arithmetic.
    metabolite_caveats: tuple = ()

    @property
    def provenance(self):
        """The weakest claim any layer makes. One illustrative layer makes the
        whole prediction illustrative."""
        if any(layer.provenance == "illustrative" for layer in self.layers):
            return "illustrative"
        return "measured"

    @property
    def peak_occupancy(self):
        return float(np.max(self.occupancy))

    @property
    def time_of_peak_occupancy_min(self):
        return float(self.times_min[int(np.argmax(self.occupancy))])

    @property
    def peak_plasma_nM(self):
        return float(np.max(self.plasma_nM))

    def layer(self, name):
        for entry in self.layers:
            if entry.name == name:
                return entry
        raise KeyError("no layer named %r in this prediction" % name)

    def occupancy_at(self, time_min):
        """Occupancy at a time on the grid this prediction was computed on.

        Linear interpolation between grid points, which is a reading of the
        stored curve and not a re-integration.
        """
        if time_min < self.times_min[0] or time_min > self.times_min[-1]:
            raise ValueError(
                "%r min is outside the predicted window %g to %g min"
                % (time_min, self.times_min[0], self.times_min[-1])
            )
        return float(np.interp(float(time_min), self.times_min, self.occupancy))

    def effects(self):
        """The D2 transducer applied pointwise to the occupancy curve.

        Applied through the drug-aware transducer. For haloperidol -- a silent
        D2 antagonist -- Emax is 1 and baseline occupancy is 0, so
        ``abs(effect)`` equals occupancy exactly: a re-expression of the
        occupancy curve on the transducer's sign convention, not a prediction of
        any behavioural or clinical quantity. The drug-aware path refuses a D2
        partial agonist (P21.5) rather than silently applying the same identity,
        because a partial agonist's occupancy overstates its effect by an
        intrinsic activity that has no single sourced human value.
        """
        return tuple(
            d2_effect_for_drug(float(value), self.drug) for value in self.occupancy
        )

    def summary(self):
        lines = [
            "%s %g mg %s in a %g kg subject, %s"
            % (self.drug, self.dose_mg, self.route, self.body_weight_kg, self.target),
            "peak plasma %.3g nM (%s), peak occupancy %.3f at %g min"
            % (
                self.peak_plasma_nM,
                self.concentration_basis,
                self.peak_occupancy,
                self.time_of_peak_occupancy_min,
            ),
            "occupancy = C / (C + %.4f nM), C measured as %s"
            % (self.ec50_nM, self.receptor_compartment),
            "provenance: %s" % self.provenance,
        ]
        for layer in self.layers:
            lines.append(
                "  %s: %s (%s) [%s]"
                % (layer.name, layer.detail, layer.provenance, layer.status)
            )
        for record in self.metabolite_caveats:
            lines.append(
                "  active metabolite %s: worst-case occupancy error %.4f if "
                "ignored, %s [%s]"
                % (
                    record.metabolite,
                    record.max_occupancy_error(),
                    "conditional on the %s affinity ratio transferring to human"
                    % record.affinity_species
                    if record.affinity_is_cross_species
                    else "human affinity ratio",
                    record.exposure_citation,
                )
            )
        return "\n".join(lines)


def default_times_min():
    """Zero to 72 hours in 15-minute steps.

    A display grid, chosen to span more than one predicted terminal half-life of
    the sourced parameter set. It is not a parameter of the model: the solver is
    given the same tolerances whatever grid it is asked to report on.
    """
    return np.linspace(0.0, 72.0 * 60.0, 289)


def _require_pk_record(route, pk_record):
    if pk_record is None:
        records = human_pk_records(_DRUG, route)
        if not records:
            raise ValueError(
                "no human %s PK record for route %r" % (_DRUG, route)
            )
        pk_record = records[0]
    if pk_record.drug != _DRUG or pk_record.species != "human":
        raise ValueError(
            "this chain is human %s only; got %s %s"
            % (_DRUG, pk_record.species, pk_record.drug)
        )
    if pk_record.route != route:
        raise ValueError(
            "PK record is for route %r, not the requested %r"
            % (pk_record.route, route)
        )
    if not pk_record.complete_for_model:
        raise ValueError(
            "the human %s %s PK record cannot drive a model: %s missing "
            "(status %r). The chain does not substitute a value for a parameter "
            "no source printed."
            % (_DRUG, route, ", ".join(pk_record.missing_fields), pk_record.status)
        )
    if pk_record.n_compartments != 2:
        raise ValueError(
            "the sourced human %s set is %r-compartment; this chain builds the "
            "two-compartment oral model P19.2 could source and will not invent "
            "the peripheral compartment a three-compartment model would need"
            % (_DRUG, pk_record.n_compartments)
        )
    if pk_record.units != "absolute":
        raise ValueError(
            "expected an absolute-unit PK record; got units=%r" % pk_record.units
        )
    return pk_record


def _require_receptor_record(receptor_record):
    if receptor_record is None:
        receptor_record = require_human_parameter(_TARGET)
    if not receptor_record.usable:
        raise ValueError(
            "the %s record cannot drive the occupancy layer: it has no human "
            "in-vivo EC50, and an in-vitro Ki is not substituted for one"
            % receptor_record.target
        )
    # Ordered before the ligand check on purpose. Both would reject a
    # risperidone record, but only this one says why the drug is unusable
    # rather than why this chain is haloperidol's. A caller who reads "the D2
    # record is for 'risperidone', not 'haloperidol'" will go and generalise
    # the chain, and the parent-only path would then accept a drug whose parent
    # concentration does not determine its occupancy (P21.4).
    require_parent_only_path(receptor_record.ligand, target=_TARGET)
    if receptor_record.ligand != _DRUG:
        raise ValueError(
            "the %s record is for %r, not %r"
            % (receptor_record.target, receptor_record.ligand, _DRUG)
        )
    if receptor_record.ec50_plasma_basis != "total":
        raise ValueError(
            "the EC50 is on a %r plasma basis but the PK layer produces total "
            "plasma concentrations; a free-fraction step would be needed and "
            "none is applied here"
            % receptor_record.ec50_plasma_basis
        )
    row = receptor_record.ec50_record
    if row is not None and row.analyte_is_summed:
        raise ValueError(
            "the %s EC50 is expressed on %r, a summed active moiety, but this "
            "chain integrates the parent drug alone. Feeding a parent "
            "concentration to a summed-moiety constant is a silent error of "
            "roughly twofold for the drugs P21.2 surveyed, so it is refused "
            "rather than approximated." % (row.drug, row.analyte)
        )
    return receptor_record


def _require_no_brain_partition_step():
    """The Kp,uu layer must still be empty for the no-partition design to hold."""
    records = kp_uu_records(_DRUG)
    if not records:
        raise ValueError("no human %s Kp,uu record to check" % _DRUG)
    record = records[0]
    if record.kp_uu is not None:
        raise ValueError(
            "a human %s Kp,uu of %r has been recorded since this chain was "
            "written. The chain identifies brain exposure from the in-vivo "
            "occupancy relation, which already contains brain penetration, so "
            "applying the partition as well would count it twice. Decide which "
            "one carries it before running."
            % (_DRUG, record.kp_uu)
        )
    return record


def _scaled_parameters(pk_record, body_weight_kg, weight_scaling):
    """Absolute V1, V2, Q and CL at `body_weight_kg`, and how they got there."""
    reference = pk_record.reference_weight_kg
    if body_weight_kg == reference:
        return (
            pk_record.V1_L,
            pk_record.V2_L,
            pk_record.Q12_L_per_h,
            pk_record.CL_L_per_h,
            "at the source's reference weight of %g kg; no scaling applied"
            % reference,
            "structural",
            pk_record.citation,
        )
    if weight_scaling is None:
        raise ValueError(
            "the source states an allometric exponent for clearance only, so "
            "predicting at %g kg rather than the reference %g kg needs a stated "
            "rule for the volumes and for Q. Pass weight_scaling=WeightScaling"
            "(volume_exponent, q_exponent, source) to supply one; it is not "
            "defaulted here because the exponents most change the predicted "
            "concentration." % (body_weight_kg, reference)
        )
    ratio = body_weight_kg / reference
    return (
        pk_record.V1_L * ratio ** weight_scaling.volume_exponent,
        pk_record.V2_L * ratio ** weight_scaling.volume_exponent,
        pk_record.Q12_L_per_h * ratio ** weight_scaling.q_exponent,
        pk_record.CL_L_per_h_at_weight(body_weight_kg),
        "scaled from %g kg to %g kg: volumes ^%g, Q ^%g (%s); clearance ^%g, "
        "which is the exponent the source printed"
        % (
            reference,
            body_weight_kg,
            weight_scaling.volume_exponent,
            weight_scaling.q_exponent,
            weight_scaling.source,
            pk_record.cl_allometric_exponent,
        ),
        "caller-supplied",
        "%s; volume and Q scaling: %s" % (pk_record.citation, weight_scaling.source),
    )


def _gate_illustrative(layers, allow_illustrative):
    """The P19.1 opt-in, applied to a whole chain rather than one constant."""
    flagged = tuple(
        layer.name for layer in layers if layer.provenance == "illustrative"
    )
    if flagged and not allow_illustrative:
        raise ValueError(
            "these chain layers are ILLUSTRATIVE (no measured human value "
            "behind them): %s; consuming them requires allow_illustrative=True "
            "and the prediction then carries provenance='illustrative'"
            % ", ".join(flagged)
        )


def _validated_times(times_min):
    times = np.asarray(times_min, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times_min must be a one-dimensional grid of at least 2 points")
    if times[0] < 0.0:
        raise ValueError("times_min must not start before the dose at t = 0")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("times_min must be strictly increasing")
    return times


def predict_occupancy(
    dose_mg,
    route="PO",
    body_weight_kg=70.0,
    times_min=None,
    pk_record=None,
    receptor_record=None,
    weight_scaling=None,
    allow_illustrative=False,
):
    """Predict a striatal D2 occupancy-time curve for an oral haloperidol dose.

    Parameters
    ----------
    dose_mg : absolute administered amount, mg. Zero is legal and means no drug.
    route : must match a sourced human PK record. Only ``"PO"`` is sourced; the
        haloperidol IV record is ``searched-and-empty``, so ``"IV"`` raises.
    body_weight_kg : defaults to the 70 kg reference weight of the PK source.
        Any other weight requires `weight_scaling`, because the source states a
        scaling rule for clearance and for nothing else.
    times_min : grid to report on, minutes from the dose. Defaults to
        :func:`default_times_min`.
    pk_record, receptor_record : supply an alternative sourced record instead of
        the registry's. Both are validated exactly as the registry's are, so a
        record with a parameter removed is refused rather than worked around.
    weight_scaling : a :class:`WeightScaling` naming the volume and Q exponents
        and where they came from.
    allow_illustrative : the P19.1 opt-in. No layer of this chain is currently
        illustrative, so the default refuses nothing today; it is the gate that
        keeps an illustrative constant from entering silently later.

    Returns
    -------
    :class:`OccupancyPrediction`, carrying the curve and the provenance of every
    layer that produced it.
    """
    dose_mg = float(dose_mg)
    if dose_mg < 0.0:
        raise ValueError("dose_mg must be non-negative; got %r" % (dose_mg,))
    body_weight_kg = float(body_weight_kg)
    if body_weight_kg <= 0.0:
        raise ValueError("body_weight_kg must be positive; got %r" % (body_weight_kg,))

    pk_record = _require_pk_record(route, pk_record)
    receptor_record = _require_receptor_record(receptor_record)
    kp_uu_record = _require_no_brain_partition_step()
    times = default_times_min() if times_min is None else _validated_times(times_min)

    v1_L, v2_L, q_L_per_h, cl_L_per_h, scaling_detail, scaling_provenance, pk_citation = (
        _scaled_parameters(pk_record, body_weight_kg, weight_scaling)
    )

    model = DosedTwoCompartmentPK(
        route=pk_record.route,
        V1_L_per_kg=v1_L / body_weight_kg,
        V2_L_per_kg=v2_L / body_weight_kg,
        Q_mL_per_min_per_kg=q_L_per_h * 1000.0 / 60.0 / body_weight_kg,
        CL_mL_per_min_per_kg=cl_L_per_h * 1000.0 / 60.0 / body_weight_kg,
        body_weight_kg=body_weight_kg,
        ka_per_min=pk_record.ka_per_min,
        molar_mass_g_per_mol=HALOPERIDOL_MW_G_PER_MOL,
        name="%s %s (Pilla Reddy 2013, apparent X/F)" % (_DRUG, pk_record.route),
        apparent_parameters=pk_record.apparent_over_F,
        species=pk_record.species,
    )
    ec50_nM = receptor_record.occupancy_constant_nM()
    # P21.1: the occupancy ceiling comes from the registry row rather than being
    # implicit in the hyperbola, because the constant and the Emax it was fitted
    # against are one measurement. For haloperidol the row records Emax pinned at
    # exactly 1, so this multiplication is the identity and the curve is
    # unchanged; the point is that a drug fitted with a free Emax cannot enter
    # this chain as if it had been fitted with Emax at 1.
    ec50_row = receptor_record.ec50_record
    emax = 1.0 if ec50_row is None else ec50_row.emax
    emax_detail = (
        "Emax is not recorded for this constant and is taken as 1"
        if ec50_row is None
        else "Emax %g, %s" % (ec50_row.emax, ec50_row.emax_convention)
    )

    layers = (
        ChainLayer(
            "pk",
            pk_record.status,
            "two-compartment oral apparent set: Vc/F %g L, Vp/F %g L, Q/F %g L/h, "
            "CL/F %g L/h, ka %g /min. F is unidentified by the source and is "
            "folded into these values, not borrowed from another study"
            % (v1_L, v2_L, q_L_per_h, cl_L_per_h, pk_record.ka_per_min),
            pk_citation,
            "measured",
        ),
        ChainLayer(
            "weight-scaling",
            "reference-weight" if weight_scaling is None else "caller-scaled",
            scaling_detail,
            pk_citation,
            scaling_provenance,
        ),
        ChainLayer(
            "exposure",
            kp_uu_record.status,
            "no brain-partition step: %s The in-vivo EC50 was fitted against "
            "plasma in living people, so brain penetration and protein binding "
            "are already inside it and the model concentration is %s"
            % (kp_uu_record.notes, model.receptor_compartment),
            kp_uu_record.citation,
            "structural",
        ),
        ChainLayer(
            "occupancy",
            receptor_record.status,
            "occupancy = Emax * C / (C + %.4f nM) on a %s plasma basis, %s. %s"
            % (
                ec50_nM,
                receptor_record.ec50_plasma_basis,
                emax_detail,
                receptor_record.ec50_uncertainty,
            ),
            receptor_record.ec50_citation,
            "fitted-in-repo",
        ),
        ChainLayer(
            "transduction",
            "structural",
            "d2_effect with Emax 1 and baseline occupancy 0, neither of which "
            "has been fitted in a human, so the effect is the occupancy curve on "
            "the transducer's sign convention and is not a behavioural claim. "
            "The identity is honest only because haloperidol is a silent D2 "
            "antagonist; the drug-aware transducer refuses a D2 partial agonist "
            "(P21.5) rather than apply Emax 1 to it",
            "neurocascade.human.d2_effect (P16.1, P21.5)",
            "structural",
        ),
    )
    _gate_illustrative(layers, allow_illustrative)

    plasma_nM = _integrate(model, dose_mg / body_weight_kg, times)
    occupancy = emax * (plasma_nM / (plasma_nM + ec50_nM))

    return OccupancyPrediction(
        drug=_DRUG,
        target=_TARGET,
        dose_mg=dose_mg,
        route=pk_record.route,
        body_weight_kg=body_weight_kg,
        times_min=times,
        plasma_nM=plasma_nM,
        occupancy=occupancy,
        concentration_basis="total plasma",
        receptor_compartment=model.receptor_compartment,
        ec50_nM=ec50_nM,
        layers=layers,
        metabolite_caveats=active_metabolites_for(_DRUG, target=_TARGET),
    )


def _integrate(model, dose_mg_per_kg, times):
    """Central-compartment concentration on `times`, in the model's own unit."""
    solution = solve_ivp(
        lambda _t, state: model.concentration_rhs(state),
        (0.0, float(times[-1])),
        model.initial_state(dose_mg_per_kg),
        t_eval=times,
        method="LSODA",
        rtol=1e-10,
        atol=1e-14,
    )
    if not solution.success:
        raise RuntimeError("PK integration failed: %s" % solution.message)
    concentration = model.receptor_concentration(solution.y)
    # LSODA can undershoot zero by a rounding-level amount on the flat tail; a
    # negative concentration is not a physical prediction and would make the
    # hyperbola non-monotonic, so it is clipped rather than reported.
    return np.clip(concentration, 0.0, None)
