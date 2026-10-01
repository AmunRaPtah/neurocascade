"""Human PK parameter records for P14.2, extended and populated in P19.2.

These are metadata records, not silently executable presets. Missing parameters
remain ``None`` so P14.4 can distinguish a sourced model from a gap.

P19.2 searched for a human haloperidol parameter set and found that the shape of
the original schema could not express what the literature actually offers. Four
distinctions were added, each because a real source needed it:

*Inapplicable is not missing.* The schema carries ``V3``/``Q13`` slots because
fentanyl needs three compartments, but human disposition is usually reported as
two. ``n_compartments`` records the count the cited source supports, and
completeness is judged against that count only. A two-compartment record with no
``V3`` is complete, not a three-compartment record with a hole in it.

*Per-kg values need a stated weight.* Human PK papers usually report absolute
volumes and clearances (L, L/h). Converting those to per-kg requires a body
weight, and a per-kg number derived from an unstated weight is untraceable.
A record declares its ``units``; the two families are never mixed, and a per-kg
value the repository derived rather than read must carry the weight it used.

*Absolute values still belong to a weight.* A population typical value applies
to a reference subject. ``reference_weight_kg`` records that subject's weight,
and ``cl_allometric_exponent`` records how the source scales clearance away from
it — 0.75 for Pilla Reddy 2013, which is not the exponent of 1 that a naive
per-kg conversion would imply.

*Apparent is not absolute.* An oral study without an intravenous arm cannot
identify F, so it reports CL/F, V/F and Q/F. Those apparent values predict the
oral plasma concentration-time course exactly, because F cancels; they do not
give absolute volumes. ``apparent_over_F`` marks a record whose disposition
values carry a hidden ``/F``, and such a record must leave ``F`` unset rather
than borrowing one from a different study.
"""
from dataclasses import dataclass

# The same six disposition slots, named in each unit family. A record populates
# exactly one family.
_DISPOSITION_FIELDS = {
    "per_kg": {
        "V1": "V1_L_per_kg", "V2": "V2_L_per_kg", "V3": "V3_L_per_kg",
        "Q12": "Q12_mL_per_min_per_kg", "Q13": "Q13_mL_per_min_per_kg",
        "CL": "CL_mL_per_min_per_kg",
    },
    "absolute": {
        "V1": "V1_L", "V2": "V2_L", "V3": "V3_L",
        "Q12": "Q12_L_per_h", "Q13": "Q13_L_per_h", "CL": "CL_L_per_h",
    },
}

# Which slots each compartment count supports. Anything outside a record's own
# entry is inapplicable to it, not missing from it.
_COMPARTMENT_SLOTS = {
    1: ("V1", "CL"),
    2: ("V1", "V2", "Q12", "CL"),
    3: ("V1", "V2", "V3", "Q12", "Q13", "CL"),
}

_ALL_DISPOSITION_FIELDS = tuple(
    name for family in _DISPOSITION_FIELDS.values() for name in family.values()
)


@dataclass(frozen=True)
class HumanPKRecord:
    drug: str
    route: str
    species: str
    V1_L_per_kg: float | None
    V2_L_per_kg: float | None
    V3_L_per_kg: float | None
    Q12_mL_per_min_per_kg: float | None
    Q13_mL_per_min_per_kg: float | None
    CL_mL_per_min_per_kg: float | None
    F: float | None
    ka_per_min: float | None
    citation: str
    status: str
    notes: str
    n_compartments: int | None = None
    units: str = "per_kg"
    apparent_over_F: bool = False
    source_reports_per_kg: bool | None = None
    reference_weight_kg: float | None = None
    cl_allometric_exponent: float | None = None
    V1_L: float | None = None
    V2_L: float | None = None
    V3_L: float | None = None
    Q12_L_per_h: float | None = None
    Q13_L_per_h: float | None = None
    CL_L_per_h: float | None = None

    def __post_init__(self):
        if self.units not in _DISPOSITION_FIELDS:
            raise ValueError(
                "%s: units=%r is not one of %s"
                % (self.drug, self.units, sorted(_DISPOSITION_FIELDS))
            )
        if self.n_compartments is not None and self.n_compartments not in _COMPARTMENT_SLOTS:
            raise ValueError(
                "%s: n_compartments=%r is not one of %s"
                % (self.drug, self.n_compartments, sorted(_COMPARTMENT_SLOTS))
            )

        own_family = set(_DISPOSITION_FIELDS[self.units].values())
        foreign = [
            name for name in _ALL_DISPOSITION_FIELDS
            if name not in own_family and getattr(self, name) is not None
        ]
        if foreign:
            raise ValueError(
                "%s: declares units=%r but populates %s from the other unit "
                "family; a record carries one unit family, never both"
                % (self.drug, self.units, ", ".join(sorted(foreign)))
            )

        populated = [name for name in own_family if getattr(self, name) is not None]
        if populated and self.units == "per_kg":
            if self.source_reports_per_kg is None:
                raise ValueError(
                    "%s: per-kg values are recorded without saying whether the "
                    "source reported per-kg or absolute units" % self.drug
                )
            if not self.source_reports_per_kg and self.reference_weight_kg is None:
                raise ValueError(
                    "%s: per-kg values were derived from absolute units without "
                    "recording the body weight used to normalise them" % self.drug
                )
        if populated and self.units == "absolute" and self.reference_weight_kg is None:
            raise ValueError(
                "%s: absolute values are population typical values and mean "
                "nothing without the reference subject's weight" % self.drug
            )

        if self.apparent_over_F and self.F is not None:
            raise ValueError(
                "%s: disposition values are apparent (X/F), so F is unidentified "
                "by this source; an F taken from elsewhere would make the record "
                "a cross-study composite without saying so" % self.drug
            )

        if self.n_compartments is not None:
            supported = _COMPARTMENT_SLOTS[self.n_compartments]
            family = _DISPOSITION_FIELDS[self.units]
            unsupported = [
                family[slot] for slot in family
                if slot not in supported and getattr(self, family[slot]) is not None
            ]
            if unsupported:
                raise ValueError(
                    "%s: a %d-compartment source cannot support %s"
                    % (self.drug, self.n_compartments, ", ".join(sorted(unsupported)))
                )

    @property
    def required_fields(self):
        """Parameters this record must carry to drive a model, given its source.

        Without a compartment count there is nothing to judge completeness
        against, so every disposition slot is demanded — the strict pre-P19.2
        behaviour, which is the right default for a record whose source has not
        been read yet.
        """
        slots = _COMPARTMENT_SLOTS.get(self.n_compartments, _COMPARTMENT_SLOTS[3])
        family = _DISPOSITION_FIELDS[self.units]
        required = [family[slot] for slot in slots]
        # F cancels out of an apparent model, which is exactly why an oral study
        # can be complete without ever having identified it.
        if not self.apparent_over_F:
            required.append("F")
        if self.route != "IV":
            required.append("ka_per_min")
        return tuple(required)

    @property
    def missing_fields(self):
        """Which required parameters are still absent, named precisely."""
        return tuple(name for name in self.required_fields if getattr(self, name) is None)

    @property
    def complete_for_model(self):
        return not self.missing_fields

    def CL_L_per_h_at_weight(self, weight_kg):
        """Clearance at a body weight, scaled the way the source scales it.

        Present so that nothing has to guess. Pilla Reddy 2013 scales CL/F by
        ``[weight / 70] ** 0.75``; treating its absolute clearance as a per-kg
        value and multiplying by weight would apply an exponent of 1 and
        over-predict clearance in anyone heavier than the reference subject.
        """
        if self.CL_L_per_h is None:
            raise ValueError("%s: no absolute clearance recorded" % self.drug)
        if self.cl_allometric_exponent is None:
            raise ValueError(
                "%s: the source states no weight-scaling rule for clearance, so "
                "this record cannot be moved off its reference weight of %s kg"
                % (self.drug, self.reference_weight_kg)
            )
        return self.CL_L_per_h * (weight_kg / self.reference_weight_kg) ** self.cl_allometric_exponent


HUMAN_PK_RECORDS = (
    HumanPKRecord(
        "fentanyl", "IV", "human", None, None, None, None, None, None, 1.0, None,
        "Shafer 1990, PMID 2248388; Scott & Stanski 1987, PMID 3100765; McClain & Hug 1980, PMID 7389247; Bae 2020, PMID 32861508",
        "parameter-set-required", "Canonical human studies establish three-compartment disposition, but this repository has not transcribed a single validated common parameter set yet.",
        n_compartments=3,
    ),
    HumanPKRecord(
        "haloperidol", "IV", "human", None, None, None, None, None, None, 1.0, None,
        "Holley 1983, PMID 6831826; Chang 1992, PMID 1579624; Li 2022, PMID 35335925; searched 2026-09-02, see docs/p19-2-human-haloperidol-pk.md",
        "searched-and-empty",
        "P19.2 searched for a human intravenous haloperidol micro-parameter set and did not find a readable one. "
        "Holley 1983 fitted open two- and three-compartment models to IV data in healthy men but printed only CL "
        "(11.8 +/- 2.9 mL/kg/min), Vss (17.8 +/- 6.5 L/kg) and the distribution half-lives; its V1/V2/Q are in the "
        "full text, which is behind a Cloudflare-gated Wiley paywall this host cannot pass. Chang 1992 states a "
        "three-compartment IV fit but published only CL, one distribution-phase volume, t1/2 and F. Li 2022 used IV "
        "data with population methods and the authors settled on one compartment. The oral record below is what "
        "the search did yield.",
        n_compartments=None,
    ),
    HumanPKRecord(
        "haloperidol", "PO", "human", None, None, None, None, None, None, None, 0.236 / 60.0,
        "Pilla Reddy V, Kozielska M, Johnson M, et al. Population pharmacokinetic-pharmacodynamic modeling of "
        "haloperidol in patients with schizophrenia using positive and negative syndrome rating scale. "
        "J Clin Psychopharmacol 2013;33(6):731-9. PMID 24113674, DOI 10.1097/JCP.0b013e3182a4ee2c. "
        "Table 2, 'Original Data Set (%RSE)' column, read from the open-access author copy at "
        "https://epub.ub.uni-muenchen.de/24066/1/oa_24066.pdf",
        "sourced-single-study-oral-apparent",
        "Single-study set, transcribed as printed: CL/F 88 L/h (%RSE 6), Q/F 233 L/h (28), Vc/F 669 L (29), "
        "Vp/F 2500 L (39), Ka 0.236 /h (18), stored here as 0.236/60 per minute. 122 subjects (20 healthy "
        "volunteers, 102 patients with schizophrenia), 538 plasma concentrations, 1-60 mg/day, oral only across "
        "all seven pooled studies. Authors: 'Haloperidol PK following oral administration was best described by a "
        "2-compartmental model with first-order absorption.' "
        "Still missing, and these are the reasons this record cannot yet drive an absolute model: (1) F was never "
        "estimated, because no arm of any pooled study was intravenous, so every volume and clearance here is "
        "apparent and V1, V2, Q12 and CL are individually unidentifiable; (2) individual body weights are not "
        "reported anywhere in the paper - 70 kg is the allometric reference weight from the fixed relationship "
        "'CL/F = TVCL * [weight / 70]0.75', not an observed mean, and the volumes carry no weight-scaling rule at "
        "all; (3) interindividual variability (IIV-CL 44.5%CV, IIV-Vc 116%CV) and the proportional residual error "
        "(0.44) are recorded in the source but not represented in this schema, so this is a typical-subject model "
        "with no uncertainty attached. "
        "Deliberately not asserted: an oral bioavailability. Holley 1983 (0.65 +/- 0.14), Cheng 1987 (0.60 +/- "
        "0.18) and Magliozzi 1985 (0.64 +/- 0.23) form a consistent cluster, and Chang 1992 reports 0.35 +/- 0.08 "
        "in Chinese patients. Dividing the apparent values above by any of those would produce a two-study "
        "composite; the acceptance criteria forbid assembling one without labelling it, and nothing downstream "
        "needs it, because F cancels out of an oral prediction.",
        n_compartments=2,
        units="absolute",
        apparent_over_F=True,
        reference_weight_kg=70.0,
        cl_allometric_exponent=0.75,
        V1_L=669.0,
        V2_L=2500.0,
        Q12_L_per_h=233.0,
        CL_L_per_h=88.0,
    ),
    HumanPKRecord(
        "diazepam", "IV", "human", None, None, None, None, None, None, 1.0, None,
        "Human diazepam PK source set pending verified transcription",
        "searched-and-empty", "No adult population parameter set is asserted here; pediatric/methodological studies are not substituted.",
    ),
)


def human_pk_records(drug=None, route=None, include_literature=True):
    """Every human PK record, hand-transcribed and bridged alike.

    P22.2 added a second source. ``HUMAN_PK_RECORDS`` above is hand-transcribed
    from the literature, one record per source read; the rows appended here come
    from :mod:`neurocascade.human.fda_pk_bridge`, which converts the FDA
    population-PK sets P21.3 transcribed out of review documents. They are
    appended at lookup time rather than typed into the tuple for two reasons.
    The first is mechanical: the bridge imports :class:`HumanPKRecord` from this
    module, so building them at module scope would be circular. The second
    matters more -- a bridged record is derived, and if it were typed into the
    tuple beside the hand-transcribed ones nothing would distinguish a number
    somebody read off a page from a number a converter produced. Every bridged
    record carries ``status="fda-population-pk-bridged"`` and names its
    application number in its citation, so the distinction survives in the data
    as well as in the import.

    The hand-transcribed rows come first, so a drug carrying both keeps the
    literature row as its default and the FDA row stays reachable by explicit
    lookup. No drug is in that position today.

    P25.2 added a third source, appended last for the same reasons: the P25.1
    journal-literature records that :mod:`neurocascade.human.literature_pk_bridge`
    converts, with ``status="literature-population-pk-bridged"``. Coming last,
    they never displace a P22-era default. ``include_literature=False`` returns
    the registry as it stood before P25.2, which the frozen P22.2 and P22.5
    results consult.
    """
    from .fda_pk_bridge import bridged_records

    records = HUMAN_PK_RECORDS + bridged_records()
    if include_literature:
        from .literature_pk_bridge import bridged_records as literature_records

        records = records + literature_records()
    if drug is not None:
        records = tuple(r for r in records if r.drug == drug)
    if route is not None:
        records = tuple(r for r in records if r.route == route)
    return records
