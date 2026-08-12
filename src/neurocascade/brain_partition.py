"""Plasma-to-brain partition: the concentration the RECEPTOR actually sees.

Why this layer has to exist
---------------------------

`pk.py` produces a *central-compartment* (plasma or blood) concentration.
Receptors sit in brain tissue, behind the blood-brain barrier, and what binds
them is the **unbound** drug in brain, not the total drug in plasma. The two
differ by two independent factors, and this codebase's occupancy ODE makes
that difference matter quantitatively rather than cosmetically:

    d(occ_i)/dt = kon_i * C_i * (1 - sum(occ)) - koff_i * occ_i

with `kon = koff / Kd` and `Kd` taken from an in vitro competition/saturation
binding assay. A binding-assay Kd is expressed in *free ligand* concentration
in the assay buffer. So `C_i` must be the free (unbound) concentration at the
receptor. Feeding it a total plasma concentration - which is what this
codebase did before P1.5 - is wrong in a direction and by a factor that are
both drug-specific.

The three quantities, and why they are not interchangeable
----------------------------------------------------------

Let C_p be total plasma concentration, C_b total brain concentration,
Cu_p unbound plasma, Cu_b unbound brain, fu_p and fu_b the unbound fractions
in plasma and in brain tissue.

    Kp     = C_b / C_p                      "brain:plasma ratio"
    Kp,uu  = Cu_b / Cu_p = Kp * fu_b / fu_p "unbound partition coefficient"

`Kp` is dominated by *tissue binding*: a lipophilic base with fu_b = 0.007
will show a large Kp (10-30) while being no better than freely permeable
across the barrier. `Kp,uu` divides that out and measures only the barrier:
1 means passive equilibrium, <1 net efflux (e.g. P-gp), >1 net uptake.

Consequences this module enforces rather than documents:

* **Kp alone cannot drive the receptor layer.** Cu_b = fu_b * Kp * C_p, so a
  Kp without an fu_b is missing exactly the factor (often 1/100 to 1/10) that
  separates "drug in the tissue" from "drug at the receptor". Multiplying a
  plasma concentration by a Kp of 22 and calling the result a driving
  concentration would overstate free drug by more than two orders of
  magnitude. `unbound_brain_factor` therefore raises for a Kp-only partition
  instead of silently substituting Kp for Kp,uu.
* **Kp,uu alone cannot be applied to a total plasma concentration either.**
  Cu_b = Kp,uu * fu_p * C_p, so a Kp,uu needs an fu_p to be usable here,
  because `pk.py` produces total plasma. `unbound_brain_factor` raises when
  fu_p is missing.

Both refusals are the same rule the rest of this codebase follows: a number
that has not been sourced is not defaulted, it is refused (see
`docs/architecture.md`, "Refusing to default").

What this layer assumes, explicitly
-----------------------------------

**Instantaneous, linear equilibrium.** Brain concentration is taken to be a
constant multiple of the current plasma concentration, with no brain state
variable of its own. That is an assumption about the *rate* of brain
distribution, and it is only legitimate when brain equilibration is fast
relative to the simulated timescale. It is not assumed blindly: each drug's
partition in `drug_pk.py` carries the published evidence for (or against) it
- for diazepam, [Friedman1986]'s brain half-life of 0.89 h against a plasma
half-life of 0.88 h; for fentanyl, [HugMurphy1981]'s "concentrations in
brain, heart, and lung equilibrated with that in plasma before 1.5 min and
declined at the same rate". Where no such evidence was found, the partition
records that.

A drug for which equilibration is *not* fast (a P-gp substrate with a long
brain equilibration half-life, an actively transported compound) needs a
brain compartment with its own state and its own influx/efflux clearances,
which is a strictly larger model than this one and needs numbers this layer
does not have. That is recorded as a limitation, not implemented on guesses.

Because the partition is linear, the whole layer reduces to one scalar per
ligand, `unbound_brain_factor`, which `engine.Simulation` multiplies into the
plasma concentration row before the occupancy ODE sees it. Keeping it a
scalar rather than a callable is deliberate: it makes the assumption
(linearity) impossible to violate accidentally, and it keeps the engine's
per-step work vectorized.

This module, like `pk.py`, contains **no drug-specific constant**. Every
number lives in `drug_pk.py` next to that drug's PK parameters, with its
citation.
"""

#: The two bases a published brain-partition number can be on. They are not
#: interchangeable; see the module docstring.
PARTITION_BASES = ("Kp", "Kp,uu")


class UnusablePartitionError(ValueError):
    """Raised when a `BrainPartition` is asked for a quantity its sourced
    numbers cannot produce - almost always an unbound brain concentration
    from a total-brain Kp with no brain unbound fraction.

    A distinct type because "this partition exists and is cited, but does not
    by itself license driving a receptor" is a *result* here, not a bug: the
    literature genuinely reports Kp far more often than Kp,uu, and the
    difference is the whole point of P1.5.
    """


class BrainPartition(object):
    """A published plasma-to-brain partition for one drug in one species.

    Parameters
    ----------
    drug, species : str
        Recorded and compared, never inferred. `species` exists so that a rat
        partition can never be silently attached to a mouse PK model: see
        `check_species`.
    basis : "Kp" or "Kp,uu"
        Which quantity `ratio` is. Required, with no default, because the two
        differ by fu_b/fu_p - a factor of 10-100 for the lipophilic bases this
        project models - and a mislabeled one is indistinguishable from an
        invented number in its effect on occupancy.
    ratio : float
        The published value, on `basis`.
    source : str
        Citation key/description. Required and non-empty.
    fu_plasma, fu_brain : float or None
        Unbound fractions in plasma and in brain tissue, in (0, 1]. Supplied
        only when they were themselves sourced; each carries its own citation
        in `source`/`note`.
    region : str or None
        Brain region the ratio was measured in, when the source resolved
        regions (fentanyl's is per-region). None means whole brain / not
        resolved.
    note : str or None
        Free text: assay caveats, equilibration evidence, disagreements with
        other studies. Kept as data so a test can assert it is still on the
        record.
    equilibration_evidence : str or None
        The published observation that licenses the instantaneous-equilibrium
        assumption for this drug (see module docstring), or None when none was
        found - in which case `assumes_rapid_equilibrium_unverified` is True
        and callers can assert on it.
    """

    def __init__(
        self,
        drug,
        species,
        basis,
        ratio,
        source,
        fu_plasma=None,
        fu_brain=None,
        region=None,
        note=None,
        equilibration_evidence=None,
    ):
        if basis not in PARTITION_BASES:
            raise ValueError(
                "basis must be one of %s (they are not interchangeable); got %r"
                % (", ".join(PARTITION_BASES), basis)
            )
        if not (ratio > 0.0):
            raise ValueError("ratio must be positive; got %r" % (ratio,))
        if not drug or not species:
            raise ValueError("drug and species are required")
        if not source:
            raise ValueError(
                "source is required: an uncited partition coefficient is "
                "exactly the invented number this codebase forbids"
            )
        for name, value in (("fu_plasma", fu_plasma), ("fu_brain", fu_brain)):
            if value is not None and not (0.0 < value <= 1.0):
                raise ValueError("%s must be in (0, 1]; got %r" % (name, value))

        self.drug = str(drug)
        self.species = str(species)
        self.basis = basis
        self.ratio = float(ratio)
        self.source = str(source)
        self.fu_plasma = None if fu_plasma is None else float(fu_plasma)
        self.fu_brain = None if fu_brain is None else float(fu_brain)
        self.region = region
        self.note = note
        self.equilibration_evidence = equilibration_evidence

    # --- the two derived coefficients --------------------------------------

    @property
    def kp(self):
        """Total brain : total plasma, or None if this partition is on the
        unbound basis and no fu's were supplied to convert back."""
        if self.basis == "Kp":
            return self.ratio
        if self.fu_plasma is None or self.fu_brain is None:
            return None
        return self.ratio * self.fu_plasma / self.fu_brain

    @property
    def kp_uu(self):
        """Unbound brain : unbound plasma, or None when the sourced numbers
        do not determine it.

        For a Kp-basis partition this is the standard identity
        Kp,uu = Kp * fu_brain / fu_plasma - a derivation from the three
        numbers supplied, not a new measurement, and only as good as the
        cross-study consistency of whichever paper each of them came from.
        """
        if self.basis == "Kp,uu":
            return self.ratio
        if self.fu_plasma is None or self.fu_brain is None:
            return None
        return self.ratio * self.fu_brain / self.fu_plasma

    # --- the factors the rest of the codebase multiplies by ----------------

    @property
    def total_brain_factor(self):
        """Multiply a TOTAL PLASMA concentration by this to get TOTAL BRAIN
        concentration. Not a receptor-driving concentration; see the module
        docstring."""
        kp = self.kp
        if kp is None:
            raise UnusablePartitionError(
                "%s/%s: a Kp,uu of %g cannot be turned into a total brain "
                "concentration without both fu_plasma and fu_brain "
                "(Kp = Kp,uu * fu_plasma / fu_brain); neither is assumed here"
                % (self.drug, self.species, self.ratio)
            )
        return kp

    @property
    def unbound_brain_factor(self):
        """Multiply a TOTAL PLASMA concentration by this to get UNBOUND BRAIN
        concentration - the quantity the receptor layer binds.

        Kp basis:     Cu_b = fu_brain * Kp * C_p          -> needs fu_brain
        Kp,uu basis:  Cu_b = Kp,uu * fu_plasma * C_p      -> needs fu_plasma

        Raises `UnusablePartitionError` rather than falling back on the other
        coefficient. Substituting Kp for Kp,uu (or skipping fu_plasma) is the
        single most consequential unsourced assumption available here.
        """
        if self.basis == "Kp":
            if self.fu_brain is None:
                raise UnusablePartitionError(
                    "%s/%s: this is a TOTAL brain:plasma ratio (Kp = %g) with "
                    "no measured brain unbound fraction, so it cannot give an "
                    "unbound brain concentration and must not drive receptor "
                    "occupancy. Cu_brain = fu_brain * Kp * C_plasma, and "
                    "fu_brain for a lipophilic base is typically 0.001-0.05, "
                    "so using Kp in its place would overstate free drug by "
                    "1-3 orders of magnitude. Source an fu_brain for this "
                    "drug and species, or a Kp,uu directly. Source of the Kp "
                    "in hand: %s" % (self.drug, self.species, self.ratio, self.source)
                )
            return self.ratio * self.fu_brain
        # basis == "Kp,uu"
        if self.fu_plasma is None:
            raise UnusablePartitionError(
                "%s/%s: Kp,uu = %g relates UNBOUND brain to UNBOUND plasma, "
                "but the PK layer produces TOTAL plasma, so an fu_plasma is "
                "required (Cu_brain = Kp,uu * fu_plasma * C_plasma). It is "
                "not defaulted. Source: %s"
                % (self.drug, self.species, self.ratio, self.source)
            )
        return self.ratio * self.fu_plasma

    @property
    def can_drive_receptor(self):
        """True when `unbound_brain_factor` is available, i.e. when this
        partition is enough to convert a plasma concentration into the
        concentration a receptor binds."""
        try:
            self.unbound_brain_factor
        except UnusablePartitionError:
            return False
        return True

    @property
    def assumes_rapid_equilibrium_unverified(self):
        """True when no published evidence of fast plasma-brain equilibration
        was found for this drug/species, so attaching this partition to a
        kinetic PK model assumes something unchecked."""
        return not self.equilibration_evidence

    # --- helpers -----------------------------------------------------------

    def check_species(self, species):
        """Raise unless `species` matches. Called by `pk.py` when a partition
        is attached to a PK model that knows its own species, so that a rat
        Kp,uu can never be applied to mouse kinetics (or vice versa) by
        accident - rule 2 of this codebase, enforced rather than trusted."""
        if species is not None and str(species) != self.species:
            raise ValueError(
                "brain partition for %s is for species %r, but it is being "
                "attached to a %r PK model; cross-species substitution of a "
                "partition coefficient is not allowed"
                % (self.drug, self.species, species)
            )

    def total_brain_concentration(self, plasma_concentration):
        return self.total_brain_factor * plasma_concentration

    def unbound_brain_concentration(self, plasma_concentration):
        return self.unbound_brain_factor * plasma_concentration

    def __repr__(self):
        bits = [
            "drug=%r" % self.drug,
            "species=%r" % self.species,
            "%s=%g" % (self.basis, self.ratio),
        ]
        if self.region:
            bits.append("region=%r" % self.region)
        if self.fu_plasma is not None:
            bits.append("fu_plasma=%g" % self.fu_plasma)
        if self.fu_brain is not None:
            bits.append("fu_brain=%g" % self.fu_brain)
        kp_uu = self.kp_uu
        if kp_uu is not None and self.basis != "Kp,uu":
            bits.append("=> Kp,uu=%.3g" % kp_uu)
        if not self.can_drive_receptor:
            bits.append("NOT receptor-drivable")
        return "BrainPartition(%s)" % ", ".join(bits)
