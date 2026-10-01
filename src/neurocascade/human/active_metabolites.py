"""Metabolites that occupy the same receptor as their parent drug (P21.4).

A chain that maps a parent plasma concentration to a receptor occupancy is
making an assumption it never states: that the parent is the only thing in the
plasma binding that receptor. For three of the drugs this repository has
touched, that assumption is false, and the FDA's own labels say so. This module
is where the assumption becomes an object with a citation on it, so that the
chain can either carry the metabolite or refuse the drug, but cannot ignore it.

Why this is not a units problem
-------------------------------

The failure mode is silent, which is what makes it worth a module. Suppose the
parent and the metabolite compete for one receptor with dissociation constants
``K_p`` and ``K_m``, and the metabolite tracks the parent at a fixed
concentration ratio ``r = C_m / C_p``. Competitive binding gives

    occupancy = (C_p/K_p + C_m/K_m) / (1 + C_p/K_p + C_m/K_m)
              = C_p (1 + d) / K_p / (1 + C_p (1 + d) / K_p)

with ``d = r * K_p / K_m``, the metabolite's share of the occupancy drive
relative to the parent's. That is *still a rectangular hyperbola in the parent
concentration alone*, with apparent constant ``K_p / (1 + d)``. So a parent-only
fit does not fail, does not look wrong, and does not warn. It returns a smaller
number and quietly relabels the metabolite's contribution as the parent's
potency.

Three consequences follow, and they are the whole of this module's logic.

1. **Within the regimen it was fitted in, a parent-only EC50 is not wrong.** It
   has absorbed the metabolite. Nothing needs to be added, and adding a
   metabolite term on top of it would count the metabolite twice.
2. **Across a change of regimen it is wrong by however much ``r`` moved.** The
   apparent constant is ``K_p / (1 + d)``; carrying it into a regimen with a
   different ``r`` misprices the drug by the ratio of the two ``(1 + d)`` terms.
   Whether ``r`` moves is an empirical question about the two regimens, not a
   modelling choice.
3. **The size of the error is bounded by ``d``, not by ``r``.** A metabolite
   present at equal concentration but 85-fold weaker contributes 1.2% of the
   drive. A metabolite at 40% of the parent's exposure and equipotent
   contributes 40%. The exposure ratio alone decides nothing.

:meth:`ActiveMetabolite.max_occupancy_error` is (3) made exact. Writing
``g = 1 + d`` and ``x = C_p / K_p``, the occupancy an ignored metabolite costs is

    x d / ((1 + g x)(1 + x))

which is maximised at ``x = 1/sqrt(g)`` and equals ``d / (1 + sqrt(g))**2``
there. That closed form is what lets a drug be cleared rather than merely
flagged: if the worst-case occupancy error over all doses is an order of
magnitude below the chain's own held-out error, the metabolite is not what is
wrong with the prediction.

What this module does not do
----------------------------

It does not implement summed-moiety occupancy for any drug, because for none of
the three registered pairs are both constants reachable. Implementing the sum
needs an in-vivo human EC50 for the parent *and* one for the metabolite, on the
same concentration basis and the same Emax convention. What the literature
actually offers is listed in ``ec50_axes_available`` per record:

* **risperidone**: moiety only. P21.2 searched for a parent-only in-vivo
  risperidone EC50 and found none anywhere; every reachable value is on the
  summed active moiety.
* **aripiprazole**: parent and moiety, but never metabolite. Kim 2012 fits the
  parent; Gruender 2008 fits both the parent and the summed moiety in the same
  subjects. Two bases are not two components, so the sum still cannot be built,
  but two bases from one study do measure something else, and
  :func:`exposure_ratio_from_ec50_bases` is what they measure.
* **haloperidol**: parent only, and the metabolite's affinity has never been
  measured in human tissue at all.

So the honest surface here is a refusal registry with the numbers that force
each refusal attached to it, plus one drug that is cleared by an arithmetic
bound rather than by assumption. Rule 10 is satisfied in the direction that
matters: the gate is not an empty container that always says no, it is three
records carrying sourced exposure ratios, half-life pairs and affinity ratios,
and those numbers are what decide each case.

Species and the rat affinity ratio
----------------------------------

Reduced haloperidol's D2 affinity has been measured in rat striatal membranes
and, for the separated enantiomers, in a medicinal-chemistry binding panel. It
has not been measured in human tissue. Rule 2 forbids substituting a species,
so the rat ratio is never used as a human number: it is stored with
``affinity_species="rat"`` and every quantity derived from it is a *stated
conditional*, reported as "conditional on the rat affinity ratio transferring"
and never folded into a prediction. The conclusion it supports is a negative
one, that the error is too small to matter, which is the direction in which a
cross-species assumption is least dangerous, and the record says so rather than
leaving the reader to notice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType

__all__ = [
    "DISPOSITIONS",
    "EXPOSURE_PROVENANCE_CLASSES",
    "EC50_AXES",
    "ActiveMetabolite",
    "ACTIVE_METABOLITES",
    "registered_parent_drugs",
    "active_metabolites_for",
    "require_parent_only_path",
    "moiety_ec50_from_components",
    "equipotency_check",
    "EquipotencyCheck",
    "exposure_ratio_from_ec50_bases",
    "GRUNDER_2008_ARIPIPRAZOLE_EC50_NG_PER_ML",
    "GRUNDER_2008_CITATION",
]


#: What the chain is allowed to do with a parent drug that has an active
#: metabolite at the same target.
#:
#: ``refused``
#:     The parent-only path is closed. Either no parent-only constant exists, or
#:     one exists but was fitted under a metabolite:parent ratio that does not
#:     hold in the regimen being predicted.
#: ``bounded``
#:     The parent-only path is open, because the metabolite's worst-case
#:     contribution has been bounded against sourced numbers and the bound is
#:     small relative to the chain's own validated error. The bound, and what it
#:     is conditional on, are recorded on the record.
DISPOSITIONS = ("refused", "bounded")

#: Provenance of the metabolite:parent exposure ratio. ``sponsor-label`` is a
#: class of its own: an approved US label is a regulator-reviewed document, but
#: the number in it is the sponsor's summary of its own studies with no primary
#: table attached, which is weaker than a primary paper and stronger than a
#: review's restatement.
EXPOSURE_PROVENANCE_CLASSES = ("primary", "sponsor-label", "second-hand")

#: Concentration axes on which an in-vivo human D2 EC50 has been published for a
#: given parent/metabolite pair.
EC50_AXES = ("parent", "metabolite", "moiety")


@dataclass(frozen=True)
class ActiveMetabolite:
    """One parent/metabolite pair at one receptor, with its disposition.

    Every numeric field is either sourced or ``None``. There is no default
    exposure ratio and no default affinity ratio, because a plausible-looking
    default here would be exactly the invented number this repository forbids:
    it would silently decide whether a drug is refused.
    """

    parent: str
    metabolite: str
    target: str

    #: One of :data:`DISPOSITIONS`.
    disposition: str
    #: Why. Required and non-empty for both dispositions: a refusal without a
    #: recorded reason is indistinguishable from a gap in the registry.
    reason: str

    #: Metabolite:parent exposure ratio, dimensionless, on :attr:`exposure_basis`.
    exposure_ratio: float | None
    #: What the ratio is a ratio *of*: AUC, steady-state trough, and so on. Two
    #: bases can differ severalfold when the half-lives differ.
    exposure_basis: str
    #: Dispersion and sample size as the source printed them, verbatim where
    #: short enough. A ratio without its spread cannot bound anything.
    exposure_spread: str
    #: The dosing regimen the ratio was measured under. A ratio is a property of
    #: a regimen, not of a drug.
    exposure_regimen: str
    exposure_citation: str
    #: One of :data:`EXPOSURE_PROVENANCE_CLASSES`.
    exposure_provenance: str

    #: K_metabolite / K_parent at :attr:`target`. Greater than 1 means the
    #: metabolite is the weaker ligand.
    affinity_ratio: float | None
    #: Species the affinity ratio was measured in. ``None`` when no ratio is
    #: recorded. Anything other than ``"human"`` makes every derived quantity a
    #: stated conditional; see :attr:`affinity_is_cross_species`.
    affinity_species: str | None
    affinity_citation: str | None

    #: Which of :data:`EC50_AXES` a published in-vivo human EC50 exists on. An
    #: empty tuple means the drug has no in-vivo occupancy relation at all.
    ec50_axes_available: tuple[str, ...]

    #: Terminal half-lives (parent, metabolite) in hours, where the source
    #: prints both. This is the evidence on whether the exposure ratio can be
    #: treated as constant across regimens; a pair that differs is a ratio that
    #: moves.
    half_life_pair_h: tuple[float, float] | None = None
    half_life_citation: str = ""

    #: Verbatim sentences from the sources that carry the load. Kept on the
    #: record rather than in prose because the refusal messages quote them, and
    #: a paraphrase of a regulatory claim is a different claim.
    quotes: tuple[str, ...] = ()

    notes: str = ""

    #: The one driven regimen in which a ``refused`` record's reason does not
    #: apply, or ``None`` when no regimen clears it. Added in P25.3 for
    #: aripiprazole, whose refusal is a regimen mismatch: the parent-only EC50
    #: was fitted after a single oral dose, so a single-dose prediction is in the
    #: regimen the constant already absorbed. A record refused for any other
    #: reason (risperidone's parent axis does not exist) leaves this ``None``.
    parent_only_regimen: str | None = None
    #: The measurement that shows the metabolite is minor in that regimen.
    parent_only_regimen_evidence: str = ""

    def __post_init__(self):
        if self.disposition not in DISPOSITIONS:
            raise ValueError(
                "disposition %r for %s/%s is not one of %s"
                % (self.disposition, self.parent, self.metabolite,
                   ", ".join(DISPOSITIONS))
            )
        if not self.reason.strip():
            raise ValueError(
                "%s/%s has an empty reason; a disposition without a recorded "
                "reason is not a record" % (self.parent, self.metabolite)
            )
        if self.exposure_provenance not in EXPOSURE_PROVENANCE_CLASSES:
            raise ValueError(
                "exposure provenance %r for %s is not one of %s"
                % (self.exposure_provenance, self.parent,
                   ", ".join(EXPOSURE_PROVENANCE_CLASSES))
            )
        for axis in self.ec50_axes_available:
            if axis not in EC50_AXES:
                raise ValueError(
                    "ec50 axis %r for %s is not one of %s"
                    % (axis, self.parent, ", ".join(EC50_AXES))
                )
        if (self.affinity_ratio is None) != (self.affinity_species is None):
            raise ValueError(
                "%s/%s records an affinity ratio without a species, or a "
                "species without a ratio; a ratio whose species is unknown "
                "cannot be reasoned about" % (self.parent, self.metabolite)
            )
        if self.exposure_ratio is not None and self.exposure_ratio <= 0.0:
            raise ValueError(
                "exposure ratio %r for %s is not positive"
                % (self.exposure_ratio, self.parent)
            )
        if self.affinity_ratio is not None and self.affinity_ratio <= 0.0:
            raise ValueError(
                "affinity ratio %r for %s is not positive"
                % (self.affinity_ratio, self.parent)
            )
        if self.parent_only_regimen is not None and (
            self.disposition != "refused" or not self.parent_only_regimen_evidence
        ):
            raise ValueError(
                "%s/%s names a clearing regimen, which only a refused record "
                "can have, and only with the evidence that clears it"
                % (self.parent, self.metabolite)
            )
        if self.disposition == "bounded" and (
            self.exposure_ratio is None or self.affinity_ratio is None
        ):
            raise ValueError(
                "%s/%s is dispositioned 'bounded' but is missing the exposure "
                "or affinity ratio the bound is computed from. A bound that "
                "cannot be computed is a refusal wearing the wrong label."
                % (self.parent, self.metabolite)
            )

    # Derived quantities.

    @property
    def affinity_is_cross_species(self):
        """True when the affinity ratio comes from a species that is not human.

        Everything derived from such a ratio is conditional on the ratio
        transferring, which rule 2 does not let this repository assume. The flag
        exists so that the conditionality is carried by the object rather than
        by whoever remembers to say it.
        """
        return self.affinity_species is not None and self.affinity_species != "human"

    def drive_ratio(self):
        """``d``: the metabolite's occupancy drive as a multiple of the parent's.

        ``d = r * K_p / K_m = exposure_ratio / affinity_ratio``. This, not the
        exposure ratio, is what decides whether a metabolite matters.
        """
        if self.exposure_ratio is None or self.affinity_ratio is None:
            raise ValueError(
                "%s/%s has no drive ratio: exposure_ratio=%r, affinity_ratio=%r. "
                "Both are required, and neither has a defensible default."
                % (self.parent, self.metabolite, self.exposure_ratio,
                   self.affinity_ratio)
            )
        return self.exposure_ratio / self.affinity_ratio

    def apparent_ec50_factor(self):
        """``1 + d``: how far a parent-only fit sits below the parent's own K.

        A parent-only EC50 fitted in the regimen this record's exposure ratio
        was measured in estimates ``K_p / (1 + d)``, not ``K_p``. The factor is
        therefore also the misprice incurred by carrying that EC50 into a
        regimen where the metabolite is absent.
        """
        return 1.0 + self.drive_ratio()

    def max_occupancy_error(self):
        """Worst-case occupancy error, over all doses, from ignoring the metabolite.

        ``d / (1 + sqrt(1 + d))**2``, the maximum over parent concentration of
        the difference between the two-ligand and parent-only hyperbolae. In
        occupancy units, so it is directly comparable with the chain's held-out
        mean absolute error.
        """
        d = self.drive_ratio()
        return d / (1.0 + math.sqrt(1.0 + d)) ** 2

    def half_life_ratio(self):
        """Metabolite:parent terminal half-life ratio, or ``None`` if unrecorded.

        A ratio far from 1 means the exposure ratio is regimen-dependent: the
        two species accumulate to steady state on different timescales, so a
        single-dose ratio and a steady-state ratio are different numbers.
        """
        if self.half_life_pair_h is None:
            return None
        parent_h, metabolite_h = self.half_life_pair_h
        return metabolite_h / parent_h

    def summary(self):
        """Multi-line human-readable record, including what each number rests on."""
        lines = [
            "%s -> %s at %s: %s" % (self.parent, self.metabolite, self.target,
                                    self.disposition),
            "  reason: %s" % self.reason,
        ]
        if self.exposure_ratio is None:
            lines.append("  exposure ratio: not published as a number")
        else:
            lines.append(
                "  exposure ratio: %.3g (%s; %s) [%s, %s]"
                % (self.exposure_ratio, self.exposure_basis, self.exposure_spread,
                   self.exposure_citation, self.exposure_provenance)
            )
        lines.append("  exposure regimen: %s" % self.exposure_regimen)
        if self.affinity_ratio is None:
            lines.append(
                "  affinity ratio: no D2 affinity measured for the metabolite"
            )
        else:
            lines.append(
                "  affinity ratio K_m/K_p: %.3g in %s [%s]%s"
                % (self.affinity_ratio, self.affinity_species,
                   self.affinity_citation,
                   ": CROSS-SPECIES, every derived number below is conditional"
                   if self.affinity_is_cross_species else "")
            )
        try:
            lines.append(
                "  drive ratio d: %.4g; parent-only EC50 sits 1/%.4g of K_p; "
                "max occupancy error from ignoring: %.4g"
                % (self.drive_ratio(), self.apparent_ec50_factor(),
                   self.max_occupancy_error())
            )
        except ValueError:
            lines.append("  drive ratio d: not computable, see above")
        ratio = self.half_life_ratio()
        if ratio is not None:
            lines.append(
                "  half-lives parent/metabolite: %g h / %g h (ratio %.2f) [%s]"
                % (self.half_life_pair_h[0], self.half_life_pair_h[1], ratio,
                   self.half_life_citation)
            )
        lines.append(
            "  in-vivo human EC50 published on: %s"
            % (", ".join(self.ec50_axes_available) or "no axis")
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

_ABILIFY = (
    "ABILIFY (aripiprazole) US label, NDA021436, openFDA effective_time "
    "20250129"
)
_RISPERDAL = (
    "RISPERDAL (risperidone) US label, NDA020272/NDA020588/NDA021444, openFDA "
    "effective_time 20260528"
)

#: Aripiprazole -> dehydro-aripiprazole.
#:
#: Refused, and the reason is a regimen mismatch rather than an absent number.
#: The only in-vivo human aripiprazole EC50 that P21.2 could reach is Kim 2012's
#: parent-only 11.1 ng/mL, fitted after a **single oral dose** in healthy young
#: men. The FDA's 40% is an explicitly **steady-state** AUC fraction. The label's
#: own half-life pair, 75 h parent against 94 h metabolite, and its own food-effect
#: Tmax pair, 3 h against 12 h, both say the metabolite lags the parent, so the
#: ratio after one dose is not the ratio at steady state and Kim's constant has
#: absorbed a smaller metabolite contribution than a steady-state prediction
#: would need. There is no published in-vivo EC50 for dehydro-aripiprazole alone
#: to reconstruct the two-ligand form from, so the mismatch cannot be corrected,
#: only refused.
#:
#: A second parent-only constant does exist, from Gründer 2008, and it is
#: steady state: 10 ng/mL in putamen and 9 ng/mL in caudate, against Kim's
#: single-dose 11.1. It does not clear the drug, for two reasons that are worth
#: separating. It is reachable second-hand only, through two reviews rather than
#: the primary. And a second parent-only constant is still a parent-only
#: constant: it has absorbed whatever metabolite was present in *its* subjects,
#: so it fixes the regimen mismatch by moving it rather than removing it.
#:
#: What Gründer supplies that nothing else does is a **moiety-basis** constant
#: for the same subjects, 20 ng/mL in putamen and 18 in caudate, which is why
#: this record's available axes are parent and moiety rather than parent alone.
#: Two bases are not two components and do not build the summed-occupancy form.
#: They do measure the exposure ratio, and without needing either affinity: see
#: :func:`exposure_ratio_from_ec50_bases`. What they measure, 1.0, is 2.5-fold
#: above the label's 0.40, and that unresolved disagreement is recorded in
#: ``notes`` as the third reason this pair is refused rather than bounded.
ARIPIPRAZOLE_DEHYDRO_ARIPIPRAZOLE = ActiveMetabolite(
    parent="aripiprazole",
    metabolite="dehydro-aripiprazole",
    target="D2",
    disposition="refused",
    reason=(
        "the only in-vivo human aripiprazole EC50 (Kim 2012, parent-only, "
        "11.1 ng/mL) was fitted after a single oral dose, while the metabolite's "
        "40% exposure share is a steady-state AUC fraction. The label's own "
        "75 h / 94 h half-life pair and 3 h / 12 h Tmax pair both show the "
        "ratio is not constant between those two regimens, so the parent-only "
        "constant has absorbed a different metabolite contribution than the "
        "prediction would need, and no dehydro-aripiprazole EC50 exists to "
        "rebuild the two-ligand form from. The one steady-state parent-only "
        "constant that does exist, Gruender 2008's 10 ng/mL in putamen, is "
        "reachable second-hand only and is itself parent-only, so it relocates "
        "the regimen mismatch rather than removing it; and its own moiety-basis "
        "companion implies a steady-state exposure ratio of 1.0, 2.5-fold above "
        "the label's 0.40, so the ratio a bound would be computed from is not "
        "settled either"
    ),
    exposure_ratio=0.40,
    exposure_basis="metabolite AUC as a fraction of parent AUC",
    exposure_spread=(
        "no dispersion published; the label states 'about 40%' with no interval "
        "and no n"
    ),
    exposure_regimen="oral, steady state",
    exposure_citation=_ABILIFY,
    exposure_provenance="sponsor-label",
    # The label asserts similar affinity but prints no metabolite Ki. Asserted
    # equipotency is not a measured ratio, so affinity_ratio stays None and the
    # assertion is carried in `quotes` where it cannot be arithmetically used.
    affinity_ratio=None,
    affinity_species=None,
    affinity_citation=None,
    # Parent from Kim 2012 and Gruender 2008; moiety from Gruender 2008 alone,
    # which two independent secondaries state is the only moiety-basis EC50
    # published for this drug.
    ec50_axes_available=("parent", "moiety"),
    half_life_pair_h=(75.0, 94.0),
    half_life_citation=_ABILIFY,
    quotes=(
        "ABILIFY activity is presumably primarily due to the parent drug, "
        "aripiprazole, and to a lesser extent, to its major metabolite, "
        "dehydro-aripiprazole, which has been shown to have affinities for D 2 "
        "receptors similar to the parent drug and represents 40% of the parent "
        "drug exposure in plasma.",
        "At steady-state, dehydro-aripiprazole, the active metabolite, "
        "represents about 40% of aripiprazole AUC in plasma.",
        "The mean elimination half-lives are about 75 hours and 94 hours for "
        "aripiprazole and dehydro-aripiprazole, respectively.",
        "Administration of a 15 mg ABILIFY tablet with a standard high-fat meal "
        "did not significantly affect the C max or AUC of aripiprazole or its "
        "active metabolite, dehydro-aripiprazole, but delayed T max by 3 hours "
        "for aripiprazole and 12 hours for dehydro-aripiprazole.",
        # The next four are from GRUNDER_2008_CITATION's reporting review, not
        # from the label. They are what puts a moiety axis on this record, and
        # the last two are what disagree with the label's 40%.
        "EC50 values for the various brain regions examined ranged from 4 to "
        "10 ng/ml, with 10 ng/ml for the putamen and 9 ng/ml for the caudate.",
        "This study is also the only one that reports EC50 estimates that are "
        "based on active moiety (aripiprazole + dehydroaripiprazole) "
        "concentrations of the drug (putamen 20 ng/ml, caudate 18 ng/ml).",
        "Thus, a not negligible fraction of total occupancy (usually 20-30%) "
        "is attributable to dehydroaripiprazole binding.",
        "Its main active metabolite, dehydroaripiprazole, has a similar "
        "receptor binding profile, and it amounts to up to 40% of the parent "
        "concentrations.",
    ),
    notes=(
        "The label prints aripiprazole's own D2 Ki (0.34 nM) and no Ki at all "
        "for dehydro-aripiprazole, so 'affinities similar to the parent drug' "
        "is an unnumbered claim. Taking it at face value gives d = 0.40 and a "
        "worst-case occupancy error of 0.084 from ignoring the metabolite, "
        "2.4x above the chain's held-out MAE of 0.0351, which is a second and "
        "independent reason this pair cannot be waved through. That figure is "
        "an illustration of the size of the problem, not a parameter: it is "
        "derived from a qualitative claim and is deliberately not stored in "
        "affinity_ratio. Gruender 2008's two bases put the same illustration "
        "higher: 20/10 in putamen and 18/9 in caudate both give an exposure "
        "ratio of 1.0 rather than 0.40 (see `exposure_ratio_from_ec50_bases`), "
        "and at d = 1.0 the worst-case error is 0.172, 4.9x the chain's MAE. "
        "The two estimates do not reconcile. Gruender's constants are printed "
        "as integers with no dispersion, but integer rounding alone bounds the "
        "putamen ratio to 0.857-1.158 and the caudate ratio to 0.842-1.176, "
        "and 0.40 is outside both. The same review states the metabolite "
        "'amounts to up to 40% of the parent concentrations' and carries "
        "'usually 20-30%' of total occupancy, and an occupancy share of 20-30% "
        "is d = 0.25-0.43, while the EC50 pair the review prints two sentences "
        "earlier implies d = 1.0, a 50% share. The exposure ratio for this drug "
        "is therefore not settled to better than 2.5-fold by the reachable "
        "sources, which is a third reason the pair is refused rather than "
        "bounded: a bound needs a ratio, and here the sources disagree about "
        "which ratio."
    ),
    # P25.3. Every reason above is about a steady-state prediction. After one
    # oral dose, the regimen Kim 2012 fitted in, the metabolite is a few percent
    # of the parent, measured in the subjects of the held-out test itself.
    parent_only_regimen="single oral dose",
    parent_only_regimen_evidence=(
        "Ito H et al. PLoS One 2012;7:e46488 (PMC3460902), Results: during the "
        "[11C]raclopride scan 3.5-4.5 h after one oral dose, dehydroaripiprazole "
        "was 0.4+/-0.2 ng/mL against parent 12.0+/-2.1 ng/mL (3 mg), 0.9+/-0.3 "
        "against 29.0+/-2.1 (6 mg) and 1.1-1.6 against 39.6-40.4 (9 mg): about "
        "3% of the parent, against the label's steady-state 40%"
    ),
)

#: Risperidone -> 9-hydroxyrisperidone (paliperidone).
#:
#: Refused, and for the strongest of the three reasons: on the parent axis the
#: quantity the chain would need does not exist in either direction. P21.2
#: searched for a parent-only in-vivo risperidone EC50 and found none anywhere;
#: every reachable value, Remington 2006's 11.06 ng/mL included, is on the summed
#: active moiety. Worse, the split between the two is CYP2D6-phenotype dependent
#: while the sum is not, which the label states outright. So parent risperidone
#: concentration is not even a well-defined function of dose across a population:
#: the same dose gives systematically different parent concentrations in
#: extensive and poor metabolisers, with the same combined moiety. A parent-only
#: path for this drug is ill-posed, which is a stronger failure than a missing
#: source: no amount of further searching would close it.
#:
#: It is also the only refused pair whose cost can be priced. The label prints
#: no ratio, but Braaten 2022 measured one in a therapeutic drug monitoring
#: cohort stratified by CYP2D6 phenotype, ten subgroups spanning 0.25 to 33.11
#: with 243 patients in the normal-metaboliser row used here, and Schotte 1995
#: measured both Ki values in the same assay against
#: the cloned human D2L. Together they give d = 12.2 and a worst-case occupancy
#: error of 0.57, sixteen times the chain's own held-out MAE. That is the
#: quantitative form of what the brief called understating the drug silently,
#: and it is why this pair is refused rather than bounded: the same arithmetic
#: that clears haloperidol at 0.0030 condemns risperidone at 0.57.
RISPERIDONE_9_HYDROXYRISPERIDONE = ActiveMetabolite(
    parent="risperidone",
    metabolite="9-hydroxyrisperidone",
    target="D2",
    disposition="refused",
    reason=(
        "no parent-only in-vivo human risperidone EC50 exists; every reachable "
        "value is on the summed active moiety (P21.2). The parent:metabolite "
        "split is additionally CYP2D6-phenotype dependent while the combined "
        "moiety is not, so parent concentration is not a well-defined function "
        "of dose across a population and a parent-only path is ill-posed rather "
        "than merely unsourced. The size is not marginal: at the normal-"
        "metaboliser ratio the metabolite carries about 12 times the parent's "
        "occupancy drive, so ignoring it costs up to 0.57 occupancy units"
    ),
    # Normal metabolisers, the largest subgroup (243 patients, 680 measurements).
    # The label declines to print any ratio, so this comes from the primary
    # literature instead.
    exposure_ratio=9.93,
    exposure_basis=(
        "metabolite:parent serum concentration ratio (MPR), 9-hydroxyrisperidone "
        "nmol/L divided by risperidone nmol/L, CYP2D6 normal metabolisers with "
        "NFIB TT"
    ),
    exposure_spread=(
        "95% CI 8.55 to 11.54 for this subgroup, but the phenotype span is the "
        "point: 33.11 (CYP2D6 ultrarapid, NFIB TC) down to 0.25 (poor "
        "metabolisers, NFIB TC), a 130-fold range across the same table. Hiemke "
        "2018 gives a further regimen split, MPR 3.6 to 22.7 (n = 168) oral "
        "against 1.2 to 4.3 (n = 30) for the long-acting injectable"
    ),
    exposure_regimen=(
        "oral, naturalistic therapeutic drug monitoring, samples 10 to 30 h "
        "after the last dose"
    ),
    exposure_citation=(
        "Braaten et al. 2022, Clin Pharmacol Ther 111(6):1266-77, PMID 35253216, "
        "human, oral; the RISPERDAL label prints no ratio at all"
    ),
    exposure_provenance="primary",
    # Metabolite Ki divided by parent Ki, so a value below 1 means the
    # metabolite binds more tightly. Same table, same radioligand, both
    # compounds named, and the receptor is the cloned human D2L rather than an
    # animal one, so unlike the haloperidol record this ratio needs no
    # cross-species step.
    affinity_ratio=4.8 / 5.9,
    affinity_species="human",
    affinity_citation=(
        "Schotte et al. 1995, Jpn J Pharmacol 69(4):399-412, Table 4, cloned "
        "human D2L in CHO cells, [3H]spiperone: risperidone Ki 5.9 nM, "
        "9-OH-risperidone Ki 4.8 nM"
    ),
    ec50_axes_available=("moiety",),
    # Extensive metabolisers. The poor-metaboliser pair is 20 h / 30 h, and the
    # fact that the pair differs so much between phenotypes while the combined
    # half-life stays near 20 h is precisely the ill-posedness above.
    half_life_pair_h=(3.0, 21.0),
    half_life_citation=_RISPERDAL + ", extensive metabolisers",
    quotes=(
        "The clinical effect from risperidone results from the combined "
        "concentrations of risperidone and its major metabolite, "
        "9-hydroxyrisperidone (paliperidone).",
        "The main metabolite, 9-hydroxyrisperidone, has similar pharmacological "
        "activity as risperidone.",
        "Although extensive metabolizers have lower risperidone and higher "
        "9-hydroxyrisperidone concentrations than poor metabolizers, the "
        "pharmacokinetics of risperidone and 9-hydroxyrisperidone combined, "
        "after single and multiple doses, are similar in extensive and poor "
        "metabolizers.",
    ),
    notes=(
        "The metabolite is paliperidone, an approved drug in its own right with "
        "its own PET occupancy study, which is the one place in this registry "
        "where a metabolite constant is reachable at all. That is what "
        "`equipotency_check` uses, and it is a test of the label's equipotency "
        "claim rather than a route to a risperidone parent constant: predicting "
        "the moiety EC50 from the metabolite's own EC50 does not recover the "
        "parent's. Plasma protein binding also differs between the two "
        "(risperidone about 90%, 9-hydroxyrisperidone 77% per the same label), "
        "so an equipotency claim on total concentrations and one on free "
        "concentrations are not the same claim. This is the one refused pair "
        "where both ratios are known, so the cost of the refusal can be priced "
        "rather than argued: d = 12.2 and the worst-case occupancy error is "
        "0.57, sixteen times the chain's held-out MAE. The conclusion does not "
        "rest on the affinity ratio, which is an in-vitro Ki ratio rather than "
        "an in-vivo potency ratio: at the same exposure ratio an affinity ratio "
        "ten times more favourable to the parent still leaves 0.197, nearly six "
        "times the MAE. It is the exposure ratio near 10 that decides this drug."
    ),
)

#: Haloperidol -> reduced haloperidol.
#:
#: The load-bearing case, because haloperidol is the one drug this repository's
#: chain is validated on. If reduced haloperidol contributed materially to D2
#: occupancy, the V1 held-out result would be measuring something other than
#: what it claims.
#:
#: It does not, but establishing that costs a cross-species step that is stated
#: rather than hidden. The human exposure ratio was measured under oral dosing:
#: Uematsu 1992
#: measured RHL/HAL AUC at 1.04 in ten patients on fixed maintenance doses. The
#: affinity ratio is rat only, and no human-tissue measurement exists. With the
#: rat 85-fold, d = 0.0122 and the worst-case occupancy error over all doses is
#: 0.0030, which is 11.5x below the chain's own held-out MAE of 0.0351. The
#: ratio would have to be wrong by more than an order of magnitude in the
#: direction of the metabolite being *stronger* than measured before this
#: mattered, and the two independent rat measurements agree that it is weaker.
HALOPERIDOL_REDUCED_HALOPERIDOL = ActiveMetabolite(
    parent="haloperidol",
    metabolite="reduced haloperidol",
    target="D2",
    disposition="bounded",
    reason=(
        "the metabolite circulates at roughly the parent's own exposure in "
        "humans (AUC ratio 1.04, Uematsu 1992) but binds D2 about 85-fold more "
        "weakly in rat striatum (Bowen 1990), giving a drive ratio of 0.012 and "
        "a worst-case occupancy error of 0.0030 over all doses, 11.5x below the "
        "chain's held-out MAE of 0.0351. The affinity ratio is rat, so this "
        "bound is a stated conditional and not a human measurement"
    ),
    exposure_ratio=1.04,
    exposure_basis="reduced-haloperidol AUC over haloperidol AUC across one day",
    exposure_spread="SD 0.48, observed range 0.39 to 1.99, n = 10 patients",
    exposure_regimen=(
        "oral, fixed maintenance 4 to 30 mg/day for more than 4 months, Japanese "
        "patients"
    ),
    exposure_citation=(
        "Uematsu et al. 1992, J Pharm Sci 81(10):1008-11, PMID 1432610, human, "
        "oral"
    ),
    exposure_provenance="primary",
    affinity_ratio=85.0,
    affinity_species="rat",
    affinity_citation=(
        "Bowen et al. 1990, Eur J Pharmacol 177(3):111-8, PMID 2155804, rat "
        "striatal membranes: reduced haloperidol Ki 239 nM against haloperidol "
        "2.8 nM"
    ),
    ec50_axes_available=("parent",),
    half_life_pair_h=None,
    quotes=(
        "The ratio of the area under the plasma concentration-time curve (AUC) "
        "of RHL for 1 day to that of HL also ranged widely from 0.39 to 1.99 "
        "(1.04 +/- 0.48, mean +/- standard deviation).",
        "reduced haloperidol bound to dopamine receptors with 85-fold lower "
        "affinity compared to haloperidol (KI = 239 nM)",
    ),
    notes=(
        "Corroboration for the affinity direction, not a second ratio: Jaen "
        "1993 (J Med Chem 36(24):3929-36, PMID 7902870) reports that both "
        "enantiomers of reduced haloperidol 'display greatly decreased affinity "
        "for D2 and D3 receptors', with affinities in the 100-200 nM range, "
        "consistent with Bowen's 239 nM. That paper is the only PubMed hit for "
        "reduced haloperidol affinity outside rat preparations and is itself a "
        "binding panel rather than human tissue. "
        "Two regimen caveats are open and neither is closed by this record. "
        "Uematsu's ratio is oral and the repository's EC50 was fitted on depot "
        "intramuscular steady state, where the ratio has not been measured; and "
        "the largest ratio Uematsu observed, 1.99, still gives only d = 0.023 "
        "and a worst-case error of 0.0057, so the bound survives the whole "
        "observed range even before the regimen question is asked. "
        "The US labels are silent on all of this: the current generic oral "
        "haloperidol label's entire CLINICAL PHARMACOLOGY section is 87 "
        "characters ('The precise mechanism of action has not been clearly "
        "established.'), and four current haloperidol decanoate ANDA labels "
        "contain no occurrence of 'metabolit' or 'reduced haloperidol' at all. "
        "That is the P20.6 finding about withdrawn innovator labels showing up "
        "again, and it is why this row is sourced from the primary literature "
        "where the other two are sourced from labels."
    ),
)


ACTIVE_METABOLITES = MappingProxyType(
    {
        "aripiprazole": (ARIPIPRAZOLE_DEHYDRO_ARIPIPRAZOLE,),
        "risperidone": (RISPERIDONE_9_HYDROXYRISPERIDONE,),
        "haloperidol": (HALOPERIDOL_REDUCED_HALOPERIDOL,),
    }
)


def registered_parent_drugs():
    """Parent drugs with at least one registered active metabolite."""
    return tuple(sorted(ACTIVE_METABOLITES))


def active_metabolites_for(drug, target="D2"):
    """Registered active metabolites of `drug` at `target`, possibly empty.

    An empty result means *not registered*, not *no active metabolite*. Absence
    of evidence is not recorded as evidence of absence anywhere in this module,
    which is why :func:`require_parent_only_path` returns records rather than a
    clearance.
    """
    records = ACTIVE_METABOLITES.get(str(drug).lower(), ())
    return tuple(r for r in records if r.target == target)


def require_parent_only_path(drug, target="D2", regimen=None):
    """Raise if `drug` may not be predicted through a parent-only chain.

    Returns the registered records for the drug, which the caller can attach to
    its provenance. The records returned are always ``bounded``; a ``refused``
    record raises, quoting the reason and the source that forces it.

    ``regimen`` names the regimen the prediction will be read under. A refused
    record whose :attr:`ActiveMetabolite.parent_only_regimen` equals it does not
    raise (P25.3); it is not returned either, because it carries no bound.
    """
    records = active_metabolites_for(drug, target=target)
    refused = [
        r
        for r in records
        if r.disposition == "refused"
        and (r.parent_only_regimen is None or r.parent_only_regimen != regimen)
    ]
    if refused:
        record = refused[0]
        raise ValueError(
            "%s cannot be predicted through the parent-only path at %s because "
            "%s is an active metabolite at the same receptor: %s. Sources: %s. "
            "Modelling the summed contribution would need an in-vivo human EC50 "
            "for each of parent and metabolite on one concentration basis, and "
            "the published axes for this drug are: %s."
            % (
                record.parent,
                record.target,
                record.metabolite,
                record.reason,
                record.exposure_citation,
                ", ".join(record.ec50_axes_available) or "none",
            )
        )
    return tuple(r for r in records if r.disposition == "bounded")


# ---------------------------------------------------------------------------
# The one place two constants exist: paliperidone is 9-hydroxyrisperidone
# ---------------------------------------------------------------------------


def moiety_ec50_from_components(
    parent_ec50_ng_per_mL,
    metabolite_ec50_ng_per_mL,
    parent_fraction_of_moiety,
):
    """EC50 of a summed moiety, from the two components' own EC50s.

    For two ligands competing at one receptor, half-occupancy is reached when
    ``C_1/K_1 + C_2/K_2 = 1``. Writing the summed moiety as ``M = C_1 + C_2``
    and the parent's share of that sum as ``w``, half-occupancy is at

        M = 1 / (w/K_1 + (1 - w)/K_2)

    which is the weighted harmonic mean of the two constants. Concentrations
    must be on one mass basis, so component EC50s expressed in molar units have
    to be converted before they are passed here: the sum a moiety assay reports
    is a sum of masses, and a harmonic mean of molar constants is a different
    number when the two molecular weights differ.
    """
    if not 0.0 <= parent_fraction_of_moiety <= 1.0:
        raise ValueError(
            "parent fraction of the moiety must lie in [0, 1], got %r"
            % (parent_fraction_of_moiety,)
        )
    for name, value in (
        ("parent", parent_ec50_ng_per_mL),
        ("metabolite", metabolite_ec50_ng_per_mL),
    ):
        if value <= 0.0:
            raise ValueError("%s EC50 must be positive, got %r" % (name, value))
    w = float(parent_fraction_of_moiety)
    return 1.0 / (w / parent_ec50_ng_per_mL + (1.0 - w) / metabolite_ec50_ng_per_mL)


@dataclass(frozen=True)
class EquipotencyCheck:
    """A predicted moiety EC50 against a measured one, with its caveats.

    This is the only comparison in P21.4 that puts a number against a number
    nobody in this repository chose, and both sides come from different groups,
    different papers and different drugs.
    """

    predicted_ng_per_mL: float
    measured_ng_per_mL: float
    measured_label: str
    measured_citation: str
    predicted_from: str
    caveats: tuple[str, ...] = ()

    @property
    def fold_error(self):
        """Measured over predicted, always reported as at least 1 in whichever
        direction it falls, so a hit is 1.00 and the sign is in the sign of
        :attr:`relative_error`."""
        ratio = self.measured_ng_per_mL / self.predicted_ng_per_mL
        return ratio if ratio >= 1.0 else 1.0 / ratio

    @property
    def relative_error(self):
        """``(predicted - measured) / measured``. Negative means under-predicted."""
        return (
            self.predicted_ng_per_mL - self.measured_ng_per_mL
        ) / self.measured_ng_per_mL


#: Molecular weights, from the Description section of each drug's US label.
RISPERIDONE_MW = 410.49
PALIPERIDONE_MW = 426.45


def equipotency_check(parent_fraction_of_moiety=0.5):
    """Test the label's equipotency claim against an independent measurement.

    The RISPERDAL label asserts that 9-hydroxyrisperidone "has similar
    pharmacological activity as risperidone". That claim is normally untestable
    here, because no parent-only risperidone EC50 exists. It becomes testable
    for one reason only: the metabolite is itself an approved drug, paliperidone,
    and has its own PET occupancy study in subjects who took paliperidone ER and
    therefore had no parent risperidone in them at all.

    So: take Arakawa's striatal paliperidone ED50, assume the label's
    equipotency on a **molar** basis, convert to the parent's mass units, and
    predict the summed-moiety ED50 that a risperidone PET study should find.
    Compare with Remington 2006, a different group, a different drug and a
    different study. The prediction is nearly independent of
    `parent_fraction_of_moiety` precisely because the two molecular weights are
    within 4% of each other, which is itself a useful property: the comparison
    does not need the moiety composition to be known.

    Every caveat that could move the answer is returned with it rather than
    discussed elsewhere; see :attr:`EquipotencyCheck.caveats`.
    """
    paliperidone_ed50 = 6.65  # ng/mL, striatum
    # Equipotency is a statement about molar affinity, so the parent's constant
    # in mass units is the metabolite's scaled by the molecular-weight ratio.
    risperidone_ed50 = paliperidone_ed50 * RISPERIDONE_MW / PALIPERIDONE_MW
    predicted = moiety_ec50_from_components(
        risperidone_ed50, paliperidone_ed50, parent_fraction_of_moiety
    )
    return EquipotencyCheck(
        predicted_ng_per_mL=predicted,
        measured_ng_per_mL=11.06,
        measured_label="risperidone active-moiety ED50, long-acting injectable",
        measured_citation=(
            "Remington et al. 2006, Am J Psychiatry 163(3):396-401, "
            "PMID 16513859; ED50 11.06 ng/mL, 95% CI 9.15 to 12.96, Emax fixed "
            "at 100%, active moiety, LAI steady state"
        ),
        predicted_from=(
            "Arakawa et al. 2008, Psychopharmacology 197(2):229-35, "
            "PMID 18058087; striatal paliperidone ED50 6.65 ng/mL, parent "
            "paliperidone only, scaled to risperidone mass units by "
            "%.2f/%.2f and combined at a parent fraction of %.2f"
            % (RISPERIDONE_MW, PALIPERIDONE_MW, parent_fraction_of_moiety)
        ),
        caveats=(
            "Arakawa 2008 is reachable second-hand only. P21.2 read 6.65 ng/mL "
            "from the sponsor's Invega medical-information page, which cites "
            "Arakawa as its reference; the primary is paywalled and its free "
            "abstract prints ED50 in dose units, not concentration. This is the "
            "weakest provenance class in the repository.",
            "The Emax convention on the Arakawa side is not established. "
            "Remington fixed Emax at 100%. P21.2 measured that this single "
            "choice moves clozapine's constant 4.6-fold and haloperidol's "
            "2.2-fold, so a convention mismatch could account for the whole "
            "discrepancy on its own.",
            "Equipotency is asserted by the sponsor and has no number attached "
            "in any label; the prediction tests the assertion rather than "
            "assuming it.",
            "Remington's value is long-acting injectable steady state. The same "
            "group's oral figure is 6.08 ng/mL, and the prediction is much "
            "closer to that, so at least part of the gap is a regimen "
            "difference rather than a failure of equipotency.",
            "Free fractions differ (risperidone about 90% bound, "
            "9-hydroxyrisperidone 77% per the RISPERDAL label), so equipotency "
            "on total concentrations and on free concentrations are different "
            "claims and this check tests only the former.",
        ),
    )


# ---------------------------------------------------------------------------
# Two bases from one study measure an exposure ratio, and need no affinities
# ---------------------------------------------------------------------------

#: Gruender 2008's D2/3 EC50 estimates for aripiprazole, ``{region: (parent,
#: moiety)}`` in ng/mL. The moiety is aripiprazole + dehydroaripiprazole. This
#: is the only study in the literature that publishes both bases for the same
#: subjects, which is what makes :func:`exposure_ratio_from_ec50_bases`
#: applicable to anything at all.
GRUNDER_2008_ARIPIPRAZOLE_EC50_NG_PER_ML = MappingProxyType(
    {"putamen": (10.0, 20.0), "caudate": (9.0, 18.0)}
)

GRUNDER_2008_CITATION = (
    "Gruender G, Fellows C, Janouschek H, Veselinovic T, Boy C, Broecheler A, "
    "et al. 'Brain and plasma pharmacokinetics of aripiprazole in patients with "
    "schizophrenia: an [18F]fallypride PET study.' Am J Psychiatry (2008) "
    "165:988-95, doi 10.1176/appi.ajp.2008.07101574; human, oral, 16 patients "
    "with schizophrenia or schizoaffective disorder at steady state on "
    "5-30 mg/day; reached second-hand through Hart, "
    "Schmitz & Gruender 2022, Front Psychiatry 13:828093, PMC9020768, and "
    "corroborated by Spangemacher et al. 2025, PMC12488982. The reviews print "
    "the values as integers with no dispersion and do not state the Emax "
    "convention of the original fit; the EC90 values they print alongside are "
    "exactly 9x the EC50, which is the constrained-model relation."
)


def exposure_ratio_from_ec50_bases(parent_ec50, moiety_ec50):
    """Metabolite:parent exposure ratio from one study's two EC50 bases.

    A study that fits occupancy against parent concentration and, in the same
    subjects, against the summed moiety, has measured the exposure ratio without
    intending to and without needing either ligand's affinity.

    At half occupancy the two-ligand condition is ``C_p/K_p + C_m/K_m = 1``. The
    parent-basis EC50 is the parent concentration at that point, ``P* = C_p*``;
    the moiety-basis EC50 is the sum at the same point, ``M* = C_p* + C_m*``. If
    the metabolite tracks the parent at ratio ``r = C_m/C_p`` then
    ``M* = C_p*(1 + r)``, so

        M* / P* = 1 + r

    exactly. Both affinities cancel: they set *where* half occupancy falls, and
    the identity divides two descriptions of the same point. So the ratio of the
    two bases is an exposure measurement, not an affinity measurement, and it is
    usable in a repository that refuses to substitute an in-vitro Ki for an
    in-vivo potency.

    Two conditions carry it, and both are conditions on the source rather than on
    the arithmetic. The two fits must be over the same subjects and scans, or the
    two constants describe different half-occupancy points and their ratio is not
    ``1 + r``. And ``r`` must be roughly constant across the subjects the fit
    pooled: if it varies, ``M*/P* - 1`` returns an exposure-weighted average of
    it, which is the right number for that cohort and not necessarily for another.

    Raises rather than returning a negative ratio when the moiety EC50 is not
    above the parent EC50. That case is not a small ratio, it is a violation of
    the identity: a sum cannot reach half occupancy at a lower value than one of
    its own parts, so the two constants are not two bases of one fit.
    """
    for name, value in (("parent", parent_ec50), ("moiety", moiety_ec50)):
        if value <= 0.0:
            raise ValueError("%s EC50 must be positive, got %r" % (name, value))
    if moiety_ec50 <= parent_ec50:
        raise ValueError(
            "moiety EC50 %r is not above the parent EC50 %r. The moiety is the "
            "parent plus something non-negative, so a moiety-basis constant "
            "below the parent-basis one means the two are not two bases of the "
            "same fit and their ratio is not an exposure ratio."
            % (moiety_ec50, parent_ec50)
        )
    return moiety_ec50 / parent_ec50 - 1.0
