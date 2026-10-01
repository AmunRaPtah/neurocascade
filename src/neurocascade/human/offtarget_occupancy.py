"""P23.2: off-target fractional occupancy from the same plasma curve as D2.

Phase 23 asks whether a drug's off-target binding *fingerprint* explains a
clinical difference that equal D2 occupancy cannot (Siafis 2023: haloperidol
11 mg/day EPS odds ratio 4.7 vs risperidone 16 mg/day 3.3, at ~87% striatal D2
occupancy both). P23.1 assembled the per-drug human Ki fingerprints at the five
named off-targets. This module turns those affinities into *occupancy-time
curves*, driven by the **same** plasma concentration-time curve that
:mod:`neurocascade.human.dose_to_occupancy` already produces for D2.

The one thing this module exists to say loudly
----------------------------------------------
An off-target occupancy computed here is a **weaker quantity** than the D2
occupancy it is plotted beside, and it is weaker for a reason the repository has
already measured. The D2 curve is driven by an **in-vivo plasma EC50** (1.885 nM,
P19.4), fitted against human PET occupancy, which absorbs plasma protein
binding, brain penetration, competition with endogenous dopamine and the tracer.
The off-target curves are driven by **in-vitro Ki** values, which absorb none of
that, applied to the same **total** plasma axis. P19.4 established for D2 that the
two are not interchangeable: the in-vivo EC50 is 1.885 nM while the in-vitro Ki
set runs 0.13-8.85 nM, so the ratio EC50/Ki spans 0.21-14.5 and *straddles
unity*. There is therefore no correction factor, not even a badly determined one,
that turns a Ki into an EC50.

D2 is the only target where both an in-vivo EC50 and an in-vitro Ki exist for
these drugs, so D2 is the only place the error of treating a Ki as an EC50 can be
*measured*. :func:`d2_invivo_vs_ki_discrepancy` does exactly that on the shared
curve: it computes the D2 occupancy the honest way (in-vivo EC50) and the way the
off-targets are forced to be computed (in-vitro Ki on total plasma), and returns
the gap. That gap -- up to ~0.37 occupancy units at clinical haloperidol
concentrations -- is the *measured floor* on how wrong every off-target occupancy
in this module could be. It is not an estimate chosen here; it is two independent
measurements (Nyberg/Regenthal PET behind the EC50, Bartuzi 2022 behind the Ki)
disagreeing.

Scope: haloperidol only, today, and not by choice
-------------------------------------------------
"The same plasma curve that drives D2" exists only for a drug the
dose-to-occupancy chain can actually drive end to end. That is haloperidol (and
olanzapine, which has no off-target Ki fingerprint from P23.1). Risperidone -- the
other half of the Siafis contrast -- *has* a P23.1 fingerprint but the chain
refuses it (parent-only disposition is ill-posed across CYP2D6 phenotype, P21.4),
and it has no in-vivo D2 EC50 here, so no plasma-curve occupancy can be produced
for it. That asymmetry is stated, not worked around: this module raises for any
drug whose plasma curve the chain does not produce.

Directional cross-check (weak, recorded as weak)
------------------------------------------------
At clinical concentrations the Ki fingerprint puts haloperidol's cortical 5-HT2A
occupancy far **below** its striatal D2 occupancy (5-HT2A Ki 25-435 nM vs D2
in-vivo EC50 1.885 nM). That ordering is the measured typical-antipsychotic
signature: haloperidol shows high striatal D2 and comparatively low cortical
5-HT2 occupancy by PET (Kapur & Remington 1996, high D2 at low dose; Farde/Nyberg
1995 established the reciprocal crossover for risperidone, 5-HT2 78-88% >= D2
75-80% at 6 mg/day). This module reproduces the *sign* of the 5-HT2A-minus-D2
occupancy difference for haloperidol. It is a one-drug, direction-only check and
is labelled as such; it is not a fitted or quantitative validation.
"""
from dataclasses import dataclass

import numpy as np

from . import offtarget_ki
from . import d2_in_vitro
from .d2_ec50 import HALOPERIDOL_D2_PLASMA_EC50_nM

# ---------------------------------------------------------------------------
# Labels. The acceptance criterion is that a Ki-derived occupancy is labelled
# distinctly from the in-vivo-fitted D2 occupancy *everywhere it appears*, so the
# labels are module constants and every record and summary line carries one.
# ---------------------------------------------------------------------------

#: The kind tag on an off-target occupancy. Distinct from QUANTITY_KIND_INVIVO.
QUANTITY_KIND_KI = "ki-derived-occupancy"

#: The kind tag on the D2 occupancy the chain already produces.
QUANTITY_KIND_INVIVO = "in-vivo-fitted-occupancy"

#: Long label carried on every Ki-derived occupancy object.
KI_DERIVED_LABEL = (
    "Ki-derived (in-vitro Ki applied to total plasma): a WEAKER quantity than "
    "the in-vivo-fitted D2 occupancy. An in-vitro Ki is not an in-vivo EC50 "
    "(P19.4); see WEAKER_THAN_D2 for the measured size of the gap."
)

#: Long label carried on the in-vivo-fitted D2 occupancy for contrast.
IN_VIVO_LABEL = (
    "in-vivo-fitted (plasma EC50 fitted on human PET occupancy, P19.4): absorbs "
    "protein binding, brain penetration and endogenous competition"
)

WEAKER_THAN_D2 = (
    "D2 is the only target with both an in-vivo EC50 (1.885 nM) and an in-vitro "
    "Ki (0.13-8.85 nM) for these drugs. The ratio EC50/Ki spans 0.21-14.5 and "
    "straddles unity, so a Ki cannot be corrected into an EC50. Every off-target "
    "occupancy here uses a Ki in the slot the D2 curve fills with an EC50, so it "
    "inherits at least that discrepancy. d2_invivo_vs_ki_discrepancy() measures "
    "it on the shared curve."
)


def ki_occupancy(concentration_nM, ki_nM):
    """Fractional occupancy ``C / (C + Ki)`` at one affinity.

    Accepts a scalar or an array of concentrations (nM). Negative concentrations
    -- which the solver can produce as a rounding-level undershoot on the flat
    tail of the plasma curve -- are clipped to zero before the hyperbola, exactly
    as :mod:`dose_to_occupancy` clips its own curve, because a negative C would
    make the hyperbola non-monotonic and is not a physical concentration.
    """
    if ki_nM <= 0.0:
        raise ValueError("ki_nM must be positive; got %r" % (ki_nM,))
    c = np.clip(np.asarray(concentration_nM, dtype=float), 0.0, None)
    occ = c / (c + ki_nM)
    return float(occ) if occ.ndim == 0 else occ


@dataclass(frozen=True)
class OffTargetOccupancyCurve:
    """A Ki-derived off-target occupancy band over a plasma-curve time grid.

    A *band*, not a point, because the P23.1 fingerprint is a Ki range (the human
    measurements of one drug at one target span up to ~17-fold). ``occ_low`` uses
    the weakest affinity (largest Ki) and ``occ_high`` the strongest (smallest
    Ki). Every instance carries :data:`QUANTITY_KIND_KI` and
    :data:`KI_DERIVED_LABEL` so the quantity cannot be read without its warning.
    """

    drug: str
    target: str
    times_min: np.ndarray
    concentration_nM: np.ndarray
    occ_low: np.ndarray
    occ_high: np.ndarray
    ki_low_nM: float
    ki_high_nM: float
    concentration_basis: str
    quantity_kind: str = QUANTITY_KIND_KI
    label: str = KI_DERIVED_LABEL

    @property
    def peak_occ_low(self):
        return float(np.max(self.occ_low))

    @property
    def peak_occ_high(self):
        return float(np.max(self.occ_high))

    def summary(self):
        return (
            "%s %s occupancy [%s]: Ki %.3g-%.4g nM on %s, peak %.3f-%.3f "
            "(%s)"
            % (
                self.drug,
                self.target,
                self.quantity_kind,
                self.ki_low_nM,
                self.ki_high_nM,
                self.concentration_basis,
                self.peak_occ_low,
                self.peak_occ_high,
                self.label,
            )
        )


def offtarget_occupancy_band(drug, target, concentration_nM):
    """``(occ_low, occ_high)`` at ``concentration_nM`` for one off-target.

    ``occ_low`` is computed at the largest human Ki in the P23.1 fingerprint and
    ``occ_high`` at the smallest, so the band is the full registered span rather
    than a point. Both are Ki-derived; see :data:`KI_DERIVED_LABEL`.
    """
    ki_low, ki_high = offtarget_ki.human_ki_range_nM(drug, target)
    # Largest Ki -> lowest occupancy, and vice versa.
    occ_low = ki_occupancy(concentration_nM, ki_high)
    occ_high = ki_occupancy(concentration_nM, ki_low)
    return occ_low, occ_high


def offtarget_occupancy_from_curve(prediction):
    """Ki-derived off-target occupancy curves from a D2 :class:`OccupancyPrediction`.

    ``prediction`` is the object :func:`dose_to_occupancy.predict_occupancy`
    returns: it already carries the plasma concentration-time curve that drives
    D2. This reuses ``prediction.plasma_nM`` on ``prediction.times_min`` -- the
    *same* curve, not a re-integration -- and returns ``{target:
    OffTargetOccupancyCurve}`` for the five Phase 23 off-targets.

    Raises for a drug that has no P23.1 off-target fingerprint, which includes
    every drug whose plasma curve the chain cannot produce (risperidone has a
    fingerprint but no curve; olanzapine has a curve but no fingerprint).
    """
    drug = prediction.drug
    if drug not in offtarget_ki.DRUGS:
        raise ValueError(
            "no P23.1 off-target fingerprint for %r; fingerprints exist for %s"
            % (drug, ", ".join(offtarget_ki.DRUGS))
        )
    concentration = np.asarray(prediction.plasma_nM, dtype=float)
    basis = prediction.receptor_compartment
    out = {}
    for target in offtarget_ki.TARGETS:
        ki_low, ki_high = offtarget_ki.human_ki_range_nM(drug, target)
        out[target] = OffTargetOccupancyCurve(
            drug=drug,
            target=target,
            times_min=np.asarray(prediction.times_min, dtype=float),
            concentration_nM=concentration,
            occ_low=ki_occupancy(concentration, ki_high),
            occ_high=ki_occupancy(concentration, ki_low),
            ki_low_nM=ki_low,
            ki_high_nM=ki_high,
            concentration_basis=basis,
        )
    return out


@dataclass(frozen=True)
class D2Discrepancy:
    """The measured floor on how weak a Ki-derived occupancy is.

    For haloperidol D2, the occupancy computed the honest way (in-vivo EC50) and
    the way the off-targets are forced to be computed (in-vitro Ki on the same
    total-plasma axis), over one concentration or array. ``difference`` is
    ``occ_invivo - occ_ki``: positive means the Ki undER-states occupancy, which
    is the sign here because the in-vivo EC50 (1.885 nM) is below the registered
    in-vitro Ki (8.85 nM).
    """

    concentration_nM: np.ndarray
    occ_invivo: np.ndarray
    occ_ki: np.ndarray
    difference: np.ndarray
    ec50_nM: float
    ki_nM: float
    ki_label: str
    invivo_label: str = IN_VIVO_LABEL
    ki_derived_label: str = KI_DERIVED_LABEL

    @property
    def max_abs_difference(self):
        return float(np.max(np.abs(np.asarray(self.difference))))


def d2_invivo_vs_ki_discrepancy(concentration_nM, ki_nM=None):
    """Measure, on the shared curve, the error of treating a Ki as an EC50.

    Haloperidol only -- it is the one target-and-drug where both an in-vivo EC50
    and an in-vitro Ki exist here. ``ki_nM`` defaults to the receptor registry's
    cited binding Ki (Bartuzi 2022, 8.85 nM); pass another of the seven P19.4
    sources to see the floor move with the denominator.
    """
    ec50 = HALOPERIDOL_D2_PLASMA_EC50_nM
    if ki_nM is None:
        record = d2_in_vitro.registry_ki()
        ki_nM = record.value
        ki_label = record.label
    else:
        ki_label = "caller-supplied"
    occ_invivo = ki_occupancy(concentration_nM, ec50)
    occ_ki = ki_occupancy(concentration_nM, ki_nM)
    difference = np.asarray(occ_invivo) - np.asarray(occ_ki)
    return D2Discrepancy(
        concentration_nM=np.asarray(concentration_nM, dtype=float),
        occ_invivo=np.asarray(occ_invivo),
        occ_ki=np.asarray(occ_ki),
        difference=difference,
        ec50_nM=ec50,
        ki_nM=ki_nM,
        ki_label=ki_label,
    )


__all__ = [
    "QUANTITY_KIND_KI",
    "QUANTITY_KIND_INVIVO",
    "KI_DERIVED_LABEL",
    "IN_VIVO_LABEL",
    "WEAKER_THAN_D2",
    "ki_occupancy",
    "OffTargetOccupancyCurve",
    "offtarget_occupancy_band",
    "offtarget_occupancy_from_curve",
    "D2Discrepancy",
    "d2_invivo_vs_ki_discrepancy",
]
