"""Human-first receptor parameter registry for P15.1-P15.4, extended in P19.4.

The original registry carried one affinity slot per target, `ki_nM`. P19.4 splits
that in two, because the two quantities that could fill it are not the same
measurement:

* An **in-vitro Ki** comes from a displacement assay against a chosen
  radioligand, in a chosen buffer, on a chosen receptor preparation. The seven
  primary human D2 haloperidol measurements collected in
  :mod:`neurocascade.human.d2_in_vitro` span 0.13 to 8.85 nM, a factor of 68, so
  there is no single number to record without the assay conditions attached.
* An **in-vivo EC50** is the plasma concentration at which half the receptors
  are occupied in a living person. It absorbs plasma protein binding, brain
  penetration, competition with endogenous dopamine, and the PET tracer the
  occupancy was measured with.

They share units, which is exactly why they get confused. The occupancy path
consumes the in-vivo EC50 and nothing else: :meth:`HumanReceptorParameter.
occupancy_constant_nM` refuses to hand back an in-vitro Ki unless the caller
passes an explicit correction factor and states where it came from.

One further correction to the original schema. A Ki and an efficacy are
properties of a **ligand at** a receptor, not of the receptor, so the record now
carries a `ligand` field. The five records still awaiting a source leave it
`None`, which is honest: no ligand has been chosen for them yet.
"""
from dataclasses import dataclass

from .d2_ec50 import d2_plasma_ec50
from .d2_in_vitro import binding_ki_fold_spread, registry_efficacy, registry_ki

# P21.1: the in-vivo EC50 and everything that qualifies it now come from one
# registry row rather than from a bare module float plus prose repeated here.
# The values below are the same floats they were; what changed is that the
# basis, the analyte and the Emax convention travel with them.
_D2_EC50 = d2_plasma_ec50("haloperidol")
_D2_KI = registry_ki()
_D2_EFFICACY = registry_efficacy()


@dataclass(frozen=True)
class HumanReceptorParameter:
    target: str
    species: str
    ki_nM: float | None
    efficacy: float | None
    evidence_context: str
    citation: str
    status: str
    notes: str
    # Fields below were added in P19.4. All are defaulted, so the records that
    # remain unsourced construct exactly as they did before.
    ligand: str | None = None
    ec50_in_vivo_nM: float | None = None
    ec50_plasma_basis: str | None = None
    ec50_origin: str | None = None
    ec50_uncertainty: str | None = None
    ec50_citation: str | None = None
    ki_radioligand: str | None = None
    ki_preparation: str | None = None
    ki_assay_conditions: str | None = None
    ki_citation: str | None = None
    efficacy_origin: str | None = None
    # Added in P21.1. The :class:`neurocascade.human.d2_ec50.D2PlasmaEC50` row
    # the EC50 fields above were copied out of, so a consumer that needs the
    # analyte or the Emax convention can reach them without this dataclass
    # growing a field per drug-specific qualifier. ``None`` on every record that
    # has no in-vivo EC50.
    ec50_record: object | None = None

    @property
    def usable(self):
        """True when this record can drive the human occupancy path.

        That requires an in-vivo EC50, not an in-vitro Ki. A record holding only
        a Ki stays unusable however well cited the Ki is.
        """
        return (
            self.species == "human"
            and self.ec50_in_vivo_nM is not None
            and self.efficacy is not None
        )

    @property
    def in_vitro_in_vivo_ratio(self):
        """In-vivo EC50 divided by in-vitro Ki, or None if either is missing.

        Reported rather than corrected for. A value far from 1 is the normal
        finding and means the two assays measured different things, not that one
        of them is wrong.
        """
        if self.ki_nM is None or self.ec50_in_vivo_nM is None or self.ki_nM == 0:
            return None
        return self.ec50_in_vivo_nM / self.ki_nM

    def occupancy_constant_nM(self, in_vitro_correction=None, correction_source=None):
        """The constant the occupancy model should use, in nM of plasma drug.

        With no arguments this returns the in-vivo EC50 and raises if there is
        none. Passing `in_vitro_correction` opts in to scaling the in-vitro Ki
        instead, and requires naming where the factor came from, so that a
        promoted Ki can never reach the model as if it were a measured in-vivo
        constant.
        """
        if in_vitro_correction is None:
            if self.ec50_in_vivo_nM is None:
                raise ValueError(
                    "no human in-vivo EC50 for %s; an in-vitro Ki will not be "
                    "substituted without an explicit correction factor" % self.target
                )
            return self.ec50_in_vivo_nM
        if self.ki_nM is None:
            raise ValueError("no in-vitro Ki recorded for %s to correct" % self.target)
        if not correction_source:
            raise ValueError(
                "correction_source must name the origin of the %r factor" % in_vitro_correction
            )
        if in_vitro_correction <= 0:
            raise ValueError("in_vitro_correction must be positive")
        return self.ki_nM * in_vitro_correction


HUMAN_RECEPTOR_PARAMETERS = (
    HumanReceptorParameter(
        "D2",
        "human",
        _D2_KI.value,
        _D2_EFFICACY.efficacy,
        "in-vivo PET",
        "Fitted in P19.4 from the P19.3 observation set; see docs/p19-4-d2-ec50-and-ki.md",
        "in-vivo-ec50-fitted",
        (
            "The Ki and the EC50 in this record are different measurements that "
            "happen to share units, and the ratio between them is %.2f. Seven "
            "primary human in-vitro Ki values were retrieved and they span %.0f-fold "
            "(0.13 to 8.85 nM), so the ratio against the in-vivo EC50 runs from 0.21 "
            "to 14.5 depending on which Ki is chosen. That range straddles unity: the "
            "in-vitro literature cannot agree on whether the in-vivo constant is "
            "larger or smaller than it is, which is why no correction factor is "
            "recorded and why occupancy_constant_nM() returns the EC50. The efficacy "
            "is negative because haloperidol is an inverse agonist at human D2, not a "
            "neutral antagonist. See neurocascade.human.d2_in_vitro for every value, "
            "its assay conditions and what each source failed to print."
            % (_D2_EC50.ec50_nM / _D2_KI.value, binding_ki_fold_spread())
        ),
        ligand=_D2_EC50.drug,
        ec50_in_vivo_nM=_D2_EC50.ec50_nM,
        ec50_plasma_basis=_D2_EC50.plasma_basis,
        ec50_origin=_D2_EC50.provenance,
        ec50_uncertainty=(
            "95%% profile-likelihood interval %.3f to %.3f nM, asymptotic standard "
            "error 0.176 nM. %s Depot intramuscular steady state, applied to an "
            "oral chain."
            % (
                _D2_EC50.interval_low_nM,
                _D2_EC50.interval_high_nM,
                _D2_EC50.interval_caveat,
            )
        ),
        ec50_citation=_D2_EC50.citation,
        ec50_record=_D2_EC50,
        ki_radioligand=_D2_KI.tracer,
        ki_preparation=_D2_KI.preparation,
        ki_assay_conditions=(
            "sodium: %s; guanine nucleotide: %s; temperature: %s; incubation: %s; "
            "nonspecific binding: %s; replicates: %s; not printed: %s"
            % (
                _D2_KI.sodium,
                _D2_KI.guanine_nucleotide,
                _D2_KI.temperature,
                _D2_KI.incubation_time,
                _D2_KI.nonspecific_binding,
                _D2_KI.replicates,
                "; ".join(_D2_KI.not_printed),
            )
        ),
        ki_citation=_D2_KI.citation,
        efficacy_origin=(
            "%s. %s, referenced to %s. Constitutive activity of the system: %s. %s "
            "The independent %s readout gives -0.83 in the same laboratory, so the "
            "measured range is -0.83 to -0.95 and no source outside that laboratory "
            "reports a number at all; Burstein 2005 agrees on the direction from its "
            "abstract only. Sign convention: %s"
            % (
                _D2_EFFICACY.classification,
                _D2_EFFICACY.value_as_printed,
                _D2_EFFICACY.reference_compound,
                _D2_EFFICACY.constitutive_activity,
                _D2_EFFICACY.citation,
                "[35S]GTPgammaS",
                "0 is neutral, negative is inverse agonism referenced to "
                "(+)-butaclamol's maximal inverse response in that assay, positive is "
                "agonism referenced to the endogenous agonist; the two ends do not "
                "share a reference",
            )
        ),
    ),
    HumanReceptorParameter("MOR", "human", None, None, "in-vivo PET", "Human PET occupancy calibration pending", "pending-human-occupancy", "No rodent affinity is promoted."),
    HumanReceptorParameter("GABA_A_BZD", "human", None, None, "in-vivo PET", "Human benzodiazepine occupancy calibration pending", "pending-human-occupancy", "No rodent affinity is promoted."),
    HumanReceptorParameter("D1", "human", None, None, "in-vitro", "Human functional affinity source not yet validated", "pending-human-source", "In-vitro values must not be treated as in-vivo occupancy parameters without an explicit correction."),
    HumanReceptorParameter("5HT2A", "human", None, None, "in-vitro", "Human functional affinity source not yet validated", "pending-human-source", "No isolated rodent rate constant is promoted."),
    HumanReceptorParameter("H1", "human", None, None, "in-vitro", "Human functional affinity source not yet validated", "pending-human-source", "Context correction remains explicit and optional."),
)


def human_receptor_parameters(target=None, evidence_context=None):
    records = HUMAN_RECEPTOR_PARAMETERS
    if target is not None:
        records = tuple(r for r in records if r.target == target)
    if evidence_context is not None:
        records = tuple(r for r in records if r.evidence_context == evidence_context)
    return records


def require_human_parameter(target):
    records = human_receptor_parameters(target)
    if not records or not records[0].usable:
        raise ValueError("no validated human receptor parameter for %s; rodent fallback is disabled" % target)
    return records[0]


def human_receptor_parameter_for(drug, target):
    """The usable record for ``drug`` at ``target``, or ``None`` if there is none.

    P21.6: the occupancy chain resolves its receptor parameter by drug now, not
    by target alone. The registry is target-keyed and today carries exactly one
    usable record per target, so this matches on the ligand and returns ``None``
    (rather than raising) when no usable record names that drug, leaving the
    caller to say which layer is missing for the drug it was actually asked for.
    """
    for record in human_receptor_parameters(target):
        if record.usable and record.ligand == drug:
            return record
    return None


def d2_occupancy_parameter_from_ec50(row):
    """A D2 occupancy record built from a drug-keyed EC50 registry row (P22.2).

    Haloperidol's :class:`HumanReceptorParameter` also carries an in-vitro Ki
    and an efficacy, sourced in P19.4. A drug added to the EC50 registry later
    carries only the in-vivo constant, and the occupancy chain consumes nothing
    else, so this record copies the row's fields and leaves the Ki and the
    efficacy ``None`` rather than borrowing haloperidol's. It is therefore not
    :attr:`~HumanReceptorParameter.usable`: that property also demands an
    efficacy, which the occupancy layer does not read.
    """
    return HumanReceptorParameter(
        "D2",
        "human",
        None,
        None,
        "in-vivo PET",
        row.citation,
        "in-vivo-ec50-transcribed",
        "Built from the %s row of neurocascade.human.d2_ec50; no in-vitro Ki "
        "and no efficacy are recorded for this drug." % row.drug,
        ligand=row.drug,
        ec50_in_vivo_nM=row.ec50_nM,
        ec50_plasma_basis=row.plasma_basis,
        ec50_origin=row.provenance,
        ec50_uncertainty=(
            "as printed by the source: %s" % row.citation
        ),
        ec50_citation=row.citation,
        ec50_record=row,
    )
