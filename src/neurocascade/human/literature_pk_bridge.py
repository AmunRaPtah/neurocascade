"""Literature population-PK records to chain-drivable PK records (P25.2).

P25.1 transcribed eight adult oral population-PK models from the journal
literature (:mod:`neurocascade.human.literature_popk`) and wired none of them
into the dose-to-occupancy chain. This module decides which of them may drive
it, in the way :mod:`neurocascade.human.fda_pk_bridge` decides for the FDA
rows. The decision is recorded in docs/p25-2-literature-pk-in-chain.md, which
was committed before this code; the rules are restated here only as far as the
code needs them.

A record converts when it is a parent-only fit, has one or two mamillary
compartments with no parallel elimination, prints clearances and volumes in L/h
and L, is apparent, names a reference weight and a defined reference subject,
has first-order absorption with a printed ka (a lag is dropped and named, any
other input is refused), and its own implied terminal half-life passed the
P25.1 label comparison (ledger V13). That last rule is the one the FDA bridge
does not have. P25.1 found that a V/F fitted to sparse trough samples is not
trustworthy for a time course, and the chain predicts time courses.

Variability and covariates are not carried: the converted record is the
source's reference subject, and :func:`dropped_parameters` names everything left
behind. ``cl_allometric_exponent`` is left unset even where a source prints one
in a covariate relation, because P25.1 stored those relations as text and did
not evaluate them; reading an exponent out of one here would be evaluating it.
A converted record therefore cannot be moved off its reference weight.

After conversion, :func:`tmax_comparisons` integrates the chain's own PK layer
for each converted record and compares the time of peak plasma concentration
with the innovator label. That is the absorption phase, which V13 (disposition
only) never tested.
"""
from dataclasses import dataclass

import numpy as np

from .fda_popk import LabelHalfLife
from .literature_popk import LITERATURE_POPPK_RECORDS, half_life_comparisons
from .pk_parameters import HumanPKRecord

#: Units each slot must be printed in. Nothing is converted.
REQUIRED_UNITS = {"CL": "L/h", "V1": "L", "V2": "L", "Q2": "L/h", "ka": "1/h", "lag": "h"}

#: Compartment counts the chain integrates for a literature record.
ALLOWED_COMPARTMENTS = (1, 2)

#: How each record's input is parameterised, mapped by hand from the printed
#: symbols: ``{(drug, first author): ((slot, printed name), ...)}`` with slots
#: ``ka`` (first-order absorption), ``lag`` (absorption lag time) and
#: ``zero-order`` (duration of a zero-order release phase). Kept here rather than
#: on the P25.1 records because it is this bridge's reading of them, not part of
#: the transcription.
INPUT_SLOTS = {
    ("clozapine", "Li"): (("ka", "Ka"),),
    ("clozapine", "Reeves"): (("ka", "Ka"),),
    ("quetiapine", "Chen"): (("ka", "Ka"),),
    ("quetiapine", "Zheng"): (("ka", "Ka"),),
    ("aripiprazole", "Koue"): (("ka", "θ2 KA"), ("lag", "θ1 ALAG")),
    ("aripiprazole", "Knights"): (("ka", "KA"),),
    ("risperidone", "Størset"): (("ka", "ka"),),
    ("paliperidone", "Korell"): (("ka", "Ka"), ("lag", "Tlag"), ("zero-order", "D1")),
}

_ENTITY = "parent"
_STATUS = "literature-population-pk-bridged"


def _author(record):
    return record.short_citation.split()[0]


def _input_slots(record):
    return dict(INPUT_SLOTS.get((record.drug, _author(record)), ()))


def _parent_half_life(record):
    for comparison in half_life_comparisons():
        if (
            comparison.drug == record.drug
            and comparison.source == record.short_citation
            and comparison.entity == _ENTITY
        ):
            return comparison
    return None


def refusals(record):
    """Why ``record`` cannot drive the chain, as a tuple; empty means it can."""
    reasons = []
    if record.route != "PO":
        reasons.append("route is %r; the chain is oral only" % record.route)
    if record.analyte != "parent":
        reasons.append(
            "analyte is %r: the parent's parameters were estimated jointly with a "
            "metabolite, and a joint fit is not collapsed into a single-analyte "
            "model (fda_pk_bridge rule 2)" % record.analyte
        )
    if record.extra_elimination_slots:
        reasons.append(
            "the parent also leaves by a parallel first-order rate constant (%s); "
            "the chain has no slot for it and folding it into clearance would "
            "produce a clearance no source printed"
            % ", ".join(printed for _, printed in record.extra_elimination_slots)
        )
    slots = dict(dict(record.disposition_slots).get(_ENTITY, ()))
    n_compartments = 2 if "Q2" in slots else 1
    if not slots:
        reasons.append("no parent disposition is mapped")
    elif n_compartments not in ALLOWED_COMPARTMENTS:
        reasons.append("a %d-compartment parent is not integrated" % n_compartments)
    inputs = _input_slots(record)
    if "ka" not in inputs:
        reasons.append("no first-order absorption rate constant is mapped")
    if "zero-order" in inputs:
        reasons.append(
            "absorption includes a zero-order release phase (%s = %g %s); the "
            "chain has a first-order depot only, and dropping a release phase "
            "reshapes the whole profile rather than shifting it"
            % (
                inputs["zero-order"],
                record.parameter(inputs["zero-order"]).value,
                record.parameter(inputs["zero-order"]).units,
            )
        )
    for slot, printed in list(slots.items()) + [
        (slot, printed) for slot, printed in inputs.items() if slot in REQUIRED_UNITS
    ]:
        units = record.parameter(printed).units
        if units != REQUIRED_UNITS[slot]:
            reasons.append(
                "%s (%r) is printed in %r and the bridge converts no units; it "
                "expects %r" % (slot, printed, units, REQUIRED_UNITS[slot])
            )
    if not record.apparent_parameters:
        reasons.append("the parameters are absolute, and the bridge handles apparent sets only")
    if record.reference_weight_kg is None:
        reasons.append(
            "the source prints no reference weight, so its typical values cannot "
            "be placed on a weight axis without choosing a subject it did not define"
        )
    comparison = _parent_half_life(record)
    if comparison is None:
        reasons.append("the parent has no P25.1 label half-life comparison, so it is unchecked")
    elif comparison.predicted_low_h != comparison.predicted_high_h:
        reasons.append(
            "no reference subject: the typical clearance depends on a covariate the "
            "source gives no reference value for, which is why P25.1 reported its "
            "half-life as the interval %.2f-%.2f h"
            % (comparison.predicted_low_h, comparison.predicted_high_h)
        )
    elif not comparison.agrees_within_twofold:
        reasons.append(
            "the implied parent half-life, %.2f h, is %.2f-fold off the label's and "
            "failed V13; a volume that fails that check is not trusted for a time "
            "course" % (comparison.predicted_low_h, comparison.fold_error)
        )
    return tuple(reasons)


def bridgeable(record):
    return not refusals(record)


def dropped_parameters(record):
    """Printed symbols the converted record does not carry, with why."""
    kept = set(dict(dict(record.disposition_slots)[_ENTITY]).values())
    inputs = _input_slots(record)
    kept.add(inputs["ka"])
    dropped = []
    for parameter in record.parameters:
        if parameter.name in kept:
            continue
        if parameter.name == inputs.get("lag"):
            reason = (
                "absorption lag; the PK model has no lag slot, so the whole "
                "profile is shifted earlier by this amount and steady-state "
                "interval statistics are unchanged"
            )
        else:
            reason = (
                "a covariate or fixed effect; the converted record is the "
                "source's reference subject only"
            )
        dropped.append((parameter.name, reason))
    return tuple(dropped)


def to_human_pk_record(record):
    """A :class:`HumanPKRecord` for ``record``, or a ``ValueError`` saying why not."""
    reasons = refusals(record)
    if reasons:
        raise ValueError(
            "%s (%s) cannot drive the dose-to-occupancy chain: %s"
            % (record.short_citation, record.drug, "; ".join(reasons))
        )
    slots = dict(dict(record.disposition_slots)[_ENTITY])

    def value(slot):
        printed = slots.get(slot)
        return None if printed is None else record.parameter(printed).value

    ka_name = _input_slots(record)["ka"]
    fixed = dict(record.fixed_parameters)
    ka_origin = (
        "ka is FIXED, not estimated from these data: %s" % fixed[ka_name]
        if ka_name in fixed
        else "ka estimated in this fit"
    )
    citation = (
        "%s. PMID %s, DOI %s. Read from %s; transcribed in P25.1 and bridged into "
        "a drivable record in P25.2. Structural model as printed: %s"
        % (record.citation, record.pmid, record.doi, record.read_from, record.structural_model)
    )
    notes = (
        "Bridged from neurocascade.human.literature_popk by literature_pk_bridge; "
        "derived, not typed. Population: %s. Reference subject: %s. %s. Apparent "
        "because: %s. Not carried: %s. Variability not represented (typical "
        "subject only): %s. Covariate relations not evaluated: %s."
        % (
            record.population,
            record.reference_subject,
            ka_origin,
            record.apparent_evidence,
            "; ".join("%s (%s)" % pair for pair in dropped_parameters(record)) or "nothing",
            "; ".join("%s %g %s" % (v.parameter, v.value, v.scale) for v in record.variability) or "none printed",
            "; ".join(c.relation for c in record.covariates) or "none",
        )
    )
    return HumanPKRecord(
        record.drug,
        "PO",
        "human",
        None, None, None, None, None, None,
        None,
        record.parameter(ka_name).value / 60.0,
        citation,
        _STATUS,
        notes,
        n_compartments=2 if "Q2" in slots else 1,
        units="absolute",
        apparent_over_F=True,
        reference_weight_kg=record.reference_weight_kg,
        cl_allometric_exponent=None,
        V1_L=value("V1"),
        V2_L=value("V2"),
        Q12_L_per_h=value("Q2"),
        CL_L_per_h=value("CL"),
    )


def bridged_records(drug=None):
    """Every P25.1 record that converts, in registry order."""
    records = LITERATURE_POPPK_RECORDS
    if drug is not None:
        records = tuple(r for r in records if r.drug == drug)
    return tuple(to_human_pk_record(r) for r in records if bridgeable(r))


def bridge_survey():
    """One line per P25.1 record: BRIDGED, or every reason it was refused."""
    return "\n".join(
        "%-12s %-14s %s"
        % (r.drug, r.short_citation, "; ".join(refusals(r)) or "BRIDGED")
        for r in LITERATURE_POPPK_RECORDS
    )


# ---------------------------------------------------------------------------
# The external check: time to peak plasma concentration against the label
# ---------------------------------------------------------------------------

#: Innovator-label Tmax statements, read from openFDA ``drug/label.json`` on
#: 2026-10-01. ``LabelHalfLife`` is reused as a container for an hours interval
#: with a quote and a source; the entity names the quantity.
LABEL_TMAX = (
    LabelHalfLife(
        drug="quetiapine",
        entity="quetiapine Tmax, oral IR",
        low_h=1.5,
        high_h=1.5,
        quote=(
            "Quetiapine is rapidly absorbed after oral administration, reaching "
            "peak plasma concentrations in 1.5 hours."
        ),
        source="NDA 020639 (SEROQUEL), SPL effective 2025-01-22, set 0584dda8-bc3c-48fe-1a90-79608f78e8a0",
        innovator=True,
    ),
    LabelHalfLife(
        drug="aripiprazole",
        entity="aripiprazole Tmax, oral tablet",
        low_h=3.0,
        high_h=5.0,
        quote=(
            "Aripiprazole is well absorbed after administration of the tablet, with "
            "peak plasma concentrations occurring within 3 hours to 5 hours"
        ),
        source="NDA 021436 (ABILIFY), SPL effective 2025-01-29, set c040bd1d-45b7-49f2-93ea-aed7220b30ac",
        innovator=True,
    ),
)

#: Twofold, the V5/V13 threshold, fixed in the design note before computing.
TMAX_FOLD_THRESHOLD = 2.0

#: The grid the chain is integrated on for the Tmax comparison: 24 h at 1 min.
_TMAX_TIMES_MIN = np.arange(0.0, 24.0 * 60.0 + 1.0, 1.0)


@dataclass(frozen=True)
class TmaxComparison:
    drug: str
    source: str
    n_compartments: int
    predicted_tmax_h: float
    label: LabelHalfLife
    ka_fixed: bool
    comment: str

    @property
    def fold_error(self):
        """Over the nearer end of the label interval; 1 inside it."""
        if self.label.contains(self.predicted_tmax_h):
            return 1.0
        if self.predicted_tmax_h > self.label.high_h:
            return self.predicted_tmax_h / self.label.high_h
        return self.label.low_h / self.predicted_tmax_h

    @property
    def agrees_within_twofold(self):
        return self.fold_error <= TMAX_FOLD_THRESHOLD


_TMAX_COMMENTS = {
    ("quetiapine", "Zheng"): (
        "ka fixed at 1.46 /h from Zhou 2015 (IR); Zhou's authors include "
        "AstraZeneca staff, who also hold the label, so the two may share data"
    ),
    ("aripiprazole", "Knights"): (
        "the sponsor's model, from the same programme as the label; the least "
        "independent comparison"
    ),
}


def predicted_tmax_h(pk_record):
    """Time of peak plasma concentration after one oral dose, in hours.

    Integrated through the chain's own PK layer at the record's reference
    weight, so it tests the code path predict() uses. Linear kinetics make it
    dose-independent; 1 mg is used.
    """
    from .dose_to_occupancy import _build_pk_model, _integrate

    weight = pk_record.reference_weight_kg
    parameters = (pk_record.V1_L, pk_record.V2_L, pk_record.Q12_L_per_h, pk_record.CL_L_per_h)
    model = _build_pk_model(pk_record, weight, None, parameters)
    plasma = _integrate(model, 1.0 / weight, _TMAX_TIMES_MIN)
    return float(_TMAX_TIMES_MIN[int(np.argmax(plasma))]) / 60.0


def label_tmax(drug):
    for item in LABEL_TMAX:
        if item.drug == drug:
            return item
    raise ValueError("no label Tmax recorded for %r" % drug)


def tmax_comparisons():
    """Every bridged literature record's Tmax against its innovator label."""
    out = []
    for record in LITERATURE_POPPK_RECORDS:
        if not bridgeable(record):
            continue
        converted = to_human_pk_record(record)
        out.append(
            TmaxComparison(
                drug=record.drug,
                source=record.short_citation,
                n_compartments=converted.n_compartments,
                predicted_tmax_h=predicted_tmax_h(converted),
                label=label_tmax(record.drug),
                ka_fixed=record.is_fixed(_input_slots(record)["ka"]),
                comment=_TMAX_COMMENTS.get((record.drug, _author(record)), ""),
            )
        )
    return tuple(out)


def tmax_table():
    lines = [
        "| drug | source | cmt | ka fixed | predicted Tmax (h) | label Tmax (h) | fold | |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in tmax_comparisons():
        label = "%g" % c.label.low_h if c.label.low_h == c.label.high_h else "%g–%g" % (
            c.label.low_h, c.label.high_h)
        lines.append(
            "| %s | %s | %d | %s | %.2f | %s | %.2f | %s |"
            % (c.drug, c.source, c.n_compartments, "yes" if c.ka_fixed else "no",
               c.predicted_tmax_h, label, c.fold_error,
               "PASS" if c.agrees_within_twofold else "FAIL")
        )
    return "\n".join(lines)
