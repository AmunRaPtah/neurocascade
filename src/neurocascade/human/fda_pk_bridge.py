"""FDA population-PK rows to chain-drivable PK records (P22.2).

P21.3 transcribed seventeen population-PK parameter sets out of FDA review
documents and stopped there, saying so explicitly: "None of this is wired into
the dose->occupancy chain... Connecting them is a later task and will require
deciding what the chain does with an apparent parameter set, which is a
modelling decision, not a transcription."

This module makes that decision. It is deliberately a separate file from
:mod:`neurocascade.human.pk_parameters`, because a bridged record is a derived
object and the difference between a hand-transcribed literature row and a row
assembled by code from a registry should be visible in the import, not buried in
a docstring.

The decision, in four parts
---------------------------

**1. An apparent set drives the chain as-is, with F left unidentified.**
``CL/F``, ``Vc/F``, ``Vp/F`` and ``Q/F`` are carried through into
``HumanPKRecord(units="absolute", apparent_over_F=True, F=None)``. This is not a
new precedent: it is exactly the treatment the hand-transcribed haloperidol row
already gets from Pilla Reddy 2013, and for the same reason. F cancels out of an
oral prediction, because the model integrates ``dose / (V/F)`` and eliminates at
``(CL/F) / (V/F)``, both of which are identified even when neither numerator nor
denominator is. Dividing by a bioavailability borrowed from a different study
would turn a single-source row into an unlabelled composite and would change the
answer, not sharpen it.

There is a second reason specific to this chain, and it matters more than the
algebra. The occupancy layer downstream is an EC50 fitted against *plasma*
concentrations measured after *oral* dosing. Those observed concentrations
already carry the true F. So the concentration this chain predicts and the
concentration the EC50 was fitted against are F-consistent with each other, and
the F-error cancels a second time at the join. What does not cancel is the
difference between the F of the population the popPK was fitted in and the F of
the population the EC50 was fitted in; that is a real and unquantified error and
it is recorded as a limitation rather than corrected.

**2. Parent-plus-metabolite fits are refused, not collapsed.**
The oral risperidone row (NDA 210655) fits risperidone and 9-hydroxyrisperidone
simultaneously with an inter-conversion rate constant. There is no honest way to
reduce that to a single-analyte two-compartment model: the parent's own
disposition is entangled with the metabolite's, and the drug's D2 occupancy is
driven by the summed moiety anyway (P21.4). The bridge refuses the row by name.

**3. A rate-constant parameterisation is refused, not converted.**
The same risperidone row prints ``kr_el,OR``, ``kr9_OR`` and ``k9_el,OR`` rather
than clearances and volumes, which is why P21.3 recorded ``mamillary_slots=()``
for it. Micro-constants and mamillary parameters are inter-convertible only when
you know which compartment each rate constant leaves, and here they do not
describe one drug's disposition at all. Refused for the same reason as (2).

**4. The absorption lag is dropped, and this is exact for what P22.2 scores.**
The LYBALVI model carries ``ALAG = 0.782 h``. The repository's PK model has no
lag-time slot. At true steady state the lag is a pure time shift of a periodic
signal, so the mean concentration over one full dosing interval, and therefore
the mean occupancy over one full dosing interval, are *exactly* unchanged by it;
so are the peak and trough values, which only move in time. The scored quantity
in P22.2 is the steady-state mean occupancy over one interval, so dropping the
lag costs nothing there. It would matter for a single-dose time course or for a
concentration at a named clock time, and :func:`dropped_parameters` names it so
that a later caller cannot lose track of the fact that it was dropped.

What the bridge refuses
-----------------------

Everything that is not a single-analyte, oral, mamillary, apparent, transcribed
row carrying a reference weight. :func:`refusals` returns the reasons as a tuple
of strings and :func:`to_human_pk_record` raises with them joined. Today exactly
one row in the registry converts -- olanzapine LYBALVI, NDA 213378 -- and
:func:`bridged_records` returns exactly that one. That is the point: the registry
holds seventeen sets and sixteen of them cannot drive this chain, which is a
finding about the FDA corpus rather than a gap in this module.
"""
from .fda_popk import FDA_POPPK_RECORDS
from .pk_parameters import HumanPKRecord

#: Units the bridge accepts for each slot, exactly as an FDA table prints them.
#: A row printing a clearance in mL/min is not converted here; it is refused, so
#: that the one place a unit conversion could go wrong stays empty until a row
#: actually needs one.
REQUIRED_UNITS = {
    "CL": "L/h",
    "V1": "L",
    "V2": "L",
    "V3": "L",
    "Q2": "L/h",
    "Q3": "L/h",
    "ka": "1/h",
}

#: Routes the oral chain can be driven with. Depot and intramuscular rows are
#: refused rather than converted: P21.3 found every depot in the registry to be
#: flip-flop, where the terminal phase is absorption rather than elimination, and
#: a mamillary disposition model fitted alongside a slow input is not a
#: disposition model this chain can reuse.
BRIDGEABLE_ROUTES = ("PO",)


def refusals(record):
    """Why ``record`` cannot drive the dose-to-occupancy chain, as a tuple.

    An empty tuple means it can. Each reason names the specific property that
    blocks it, so that a caller reading the refusal learns something about the
    source rather than being told to try a different row.
    """
    reasons = []
    if record.status != "transcribed":
        reasons.append(
            "status is %r, not 'transcribed' (barrier: %s)"
            % (record.status, record.extraction_barrier)
        )
        # Nothing below can be checked on a row with no parameters.
        return tuple(reasons)
    if record.route not in BRIDGEABLE_ROUTES:
        reasons.append(
            "route is %r; this chain is oral only and does not reuse a "
            "disposition model fitted alongside a depot input" % record.route
        )
    if record.analyte != "parent":
        reasons.append(
            "analyte is %r; a fit that carries a metabolite cannot be reduced to "
            "a single-analyte disposition model without inventing the split"
            % record.analyte
        )
    if not record.is_mamillary:
        reasons.append(
            "the fitted model is not a mamillary one (%s), so it has no "
            "clearance-and-volume mapping to convert" % record.structural_model
        )
    if not record.apparent_parameters:
        reasons.append(
            "the parameters are absolute rather than apparent; the bridge has "
            "only decided what to do with an apparent oral set, and an absolute "
            "one would need its bioavailability handled explicitly"
        )
    if record.reference_weight_kg is None:
        reasons.append(
            "the review never printed the weight of the reference subject the "
            "typical values belong to, so they cannot be placed on a weight axis"
        )
    slots = dict(record.mamillary_slots)
    absorption = dict(record.absorption_slots)
    if record.route != "IV" and "ka" not in absorption:
        reasons.append(
            "no absorption rate constant is mapped, and an oral model needs one"
        )
    for slot, printed in list(slots.items()) + list(absorption.items()):
        if slot not in REQUIRED_UNITS:
            continue
        parameter = record.parameter(printed)
        if parameter.units is None:
            reasons.append(
                "%s (%r) was printed without a unit, and the unit is not inferred"
                % (slot, printed)
            )
        elif parameter.units != REQUIRED_UNITS[slot]:
            reasons.append(
                "%s (%r) is printed in %r, and the bridge converts no units; it "
                "expects %r" % (slot, printed, parameter.units, REQUIRED_UNITS[slot])
            )
    return tuple(reasons)


def bridgeable(record):
    """True when :func:`to_human_pk_record` would succeed on ``record``."""
    return not refusals(record)


def dropped_parameters(record):
    """Printed symbols the conversion does not carry, with why, as a tuple.

    Covariate effects are dropped because :class:`HumanPKRecord` represents a
    typical subject and has no covariate slots; the lag time is dropped because
    the model has no lag slot. Both are returned rather than silently discarded,
    so that a caller comparing a bridged prediction against a source's own
    simulation can see the whole difference between them.
    """
    kept = set(dict(record.mamillary_slots).values())
    kept |= {
        printed for slot, printed in record.absorption_slots if slot != "lag"
    }
    kept |= set(dict(record.allometric_slots).values())
    dropped = []
    for parameter in record.parameters:
        if parameter.name in kept:
            continue
        if parameter.name in dict(record.absorption_slots).values():
            dropped.append(
                (parameter.name,
                 "the PK model has no lag-time slot; at steady state a lag is a "
                 "pure time shift of a periodic signal, so the mean, peak and "
                 "trough concentrations over one dosing interval are unchanged "
                 "by dropping it, and only their timing moves")
            )
        else:
            dropped.append(
                (parameter.name,
                 "a covariate effect; HumanPKRecord carries a typical subject "
                 "and has no covariate slots, so this row predicts for the "
                 "reference subject only")
            )
    return tuple(dropped)


def to_human_pk_record(record):
    """A :class:`HumanPKRecord` for ``record``, or a ``ValueError`` saying why not.

    The returned record is ``units="absolute"``, ``apparent_over_F=True`` and
    ``F=None``: see the module docstring for why F is left unidentified rather
    than borrowed.
    """
    reasons = refusals(record)
    if reasons:
        raise ValueError(
            "%s %s (%s, %s) cannot drive the dose-to-occupancy chain: %s"
            % (
                record.drug,
                record.application_number,
                record.formulation,
                record.analyte,
                "; ".join(reasons),
            )
        )
    slots = dict(record.mamillary_slots)
    absorption = dict(record.absorption_slots)
    allometric = dict(record.allometric_slots)

    def value(mapping, slot):
        printed = mapping.get(slot)
        if printed is None:
            return None
        return record.parameter(printed).value

    cl_exponent = value(allometric, "cl_weight_exponent")
    ka_per_h = value(absorption, "ka")
    citation = (
        "%s, %s %s, %s, reviewed %s, transcribed in P21.3 and bridged into a "
        "drivable record in P22.2. Structural model as the review states it: %s. "
        "Apparent because %s. Reference subject: %s"
        % (
            record.drug,
            record.formulation,
            record.route,
            record.application_number,
            record.review_date,
            record.structural_model,
            record.apparent_evidence,
            record.reference_subject or "not described by the review",
        )
    )
    dropped = dropped_parameters(record)
    notes = (
        "Bridged from the FDA population-PK registry by "
        "neurocascade.human.fda_pk_bridge; the transcription of record is "
        "fda_popk.record(%r), and this object is derived from it rather than "
        "typed alongside it. Population: %s. %s. "
        "Every disposition value is apparent (X/F): the review never identified "
        "an oral bioavailability, so CL, V1, V2 and Q are individually "
        "unidentifiable and only the ratios the model actually uses are "
        "determined. Not carried across: %s. "
        "Inter-individual variability is recorded on the FDA row and is not "
        "represented in this schema, so this is a typical-subject model with no "
        "uncertainty attached."
        % (
            record.application_number,
            record.population,
            record.notes,
            "; ".join("%s (%s)" % pair for pair in dropped) or "nothing",
        )
    )
    return HumanPKRecord(
        record.drug,
        record.route,
        "human",
        None, None, None, None, None, None,
        None,
        None if ka_per_h is None else ka_per_h / 60.0,
        citation,
        "fda-population-pk-bridged",
        notes,
        n_compartments=record.n_compartments,
        units="absolute",
        apparent_over_F=True,
        reference_weight_kg=record.reference_weight_kg,
        cl_allometric_exponent=cl_exponent,
        V1_L=value(slots, "V1"),
        V2_L=value(slots, "V2"),
        V3_L=value(slots, "V3"),
        Q12_L_per_h=value(slots, "Q2"),
        Q13_L_per_h=value(slots, "Q3"),
        CL_L_per_h=value(slots, "CL"),
    )


def bridged_records(drug=None):
    """Every FDA row that converts, as :class:`HumanPKRecord` objects."""
    records = FDA_POPPK_RECORDS
    if drug is not None:
        records = tuple(r for r in records if r.drug == drug)
    return tuple(
        to_human_pk_record(r) for r in records if bridgeable(r)
    )


def bridge_survey():
    """One line per FDA row: converted, or the first reason it was not.

    The honest summary of what the FDA corpus can and cannot drive. Printed in
    docs/p22-2-label-dose-test.md rather than described.
    """
    lines = []
    for record in FDA_POPPK_RECORDS:
        reasons = refusals(record)
        lines.append(
            "%-12s %-16s %-3s %-22s %s"
            % (
                record.drug,
                record.application_number,
                record.route,
                record.analyte,
                "BRIDGED" if not reasons else reasons[0],
            )
        )
    return "\n".join(lines)
