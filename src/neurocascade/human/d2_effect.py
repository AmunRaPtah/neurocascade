"""D2 occupancy-limited effect model (P16.1; partial-agonism guard, P21.5).

The transducer maps a D2 receptor occupancy onto a scalar effect. With
``emax = 1`` and ``baseline_occupancy = 0`` the effect is ``-occupancy``
exactly, so ``abs(effect) == occupancy``. That identity is honest *only for a
silent D2 antagonist*: every occupied receptor is a receptor removed from
dopamine signalling, so occupancy is the fractional loss of D2 tone, and the
transducer is a sign-convention re-expression of the occupancy curve rather
than a prediction.

Why a partial agonist breaks that identity
------------------------------------------
A D2 partial agonist (aripiprazole) occupies the receptor but itself produces
D2 signalling, so occupancy *overstates* the functional effect by the drug's
intrinsic activity. Applying ``emax = 1`` to a partial agonist would silently
assert that 90% occupancy means 90% loss of D2 tone -- the very claim that
aripiprazole's clinic falsifies: it sustains ~85-95% striatal D2 occupancy yet
is largely extrapyramidal-sparing, unlike a full antagonist at that occupancy
([Mamo2007] PMID 17728427; [Yokoi2002] PMID 12093598; [Grunder2008]
PMID 18381901).

There is no single sourced intrinsic activity that would let the transducer
represent aripiprazole honestly. The value is irreducibly system-dependent:

* [Burris2002] Burris et al., J Pharmacol Exp Ther 302(1):381-389, PMID
  12065741, DOI 10.1124/jpet.102.033175 -- in one human-D2L/CHO cAMP assay,
  "The efficacy of aripiprazole relative to that of dopamine varied from 25%
  in cells that lacked spare receptors for dopamine to 90% in cells with
  receptor reserve." One assay, one cell line: 25% to 90% by receptor reserve
  alone.
* [MailmanMurthy2010] Mailman & Murthy, Curr Pharm Des 16(5):488-501, PMID
  19909227 -- aripiprazole "appears to elicit D2-mediated functional effects
  that encompass the whole range of classic pharmacological intrinsic
  activity" (pure antagonist at GTPgammaS/GIRK, full agonist for tyrosine
  hydroxylase), and its cAMP intrinsic activity is cell-line-dependent.
* No in-vivo human D2 intrinsic-activity fraction has ever been measured
  (searched 2026-09-08, PubMed and Europe PMC; not found). Intrinsic activity
  is an in-vitro operational construct -- an Emax "relative to a reference
  agonist in any single given experimental system" ([MailmanMurthy2010]) --
  not a quantity measured as one number in humans.

So the transducer does not invent an Emax for a partial agonist. The
drug-aware entry point (:func:`d2_effect_for_drug`) refuses one by name,
naming the reason, rather than letting ``emax = 1`` pass silently. This is the
P21.5 acceptance criterion: the occupancy-equals-effect identity no longer
holds silently for a partial agonist.
"""
from dataclasses import dataclass
from types import MappingProxyType


class PartialAgonistTransductionError(ValueError):
    """The occupancy-limited transducer was asked to act on a D2 partial agonist.

    A subclass of :class:`ValueError` so existing broad handlers still catch it,
    but named so the refusal is legible: a partial agonist's occupancy does not
    map to effect by any single sourced intrinsic activity.
    """


@dataclass(frozen=True)
class D2Effect:
    occupancy: float
    baseline_occupancy: float = 0.0
    emax: float = 1.0
    direction: float = -1.0

    def __post_init__(self):
        for name, value in (("occupancy", self.occupancy), ("baseline_occupancy", self.baseline_occupancy)):
            if not 0.0 <= value <= 1.0:
                raise ValueError("%s must be in [0, 1]" % name)
        if self.emax < 0.0:
            raise ValueError("emax must be non-negative")
        if self.direction not in (-1.0, 1.0):
            raise ValueError("direction must be -1 or 1")

    @property
    def effect(self):
        """Drug-induced effect relative to baseline, bounded by Emax."""
        return self.direction * self.emax * max(0.0, self.occupancy - self.baseline_occupancy)

    @property
    def available_reserve(self):
        return max(0.0, 1.0 - self.baseline_occupancy)


@dataclass(frozen=True)
class D2FunctionalClass:
    """A sourced statement of a D2 ligand's functional type for the transducer.

    ``kind`` is either ``"silent_antagonist"`` (occupancy maps to effect with no
    intrinsic-activity term, so the occupancy identity is honest) or
    ``"partial_agonist"`` (occupancy overstates effect by an intrinsic activity
    with no single sourced value, so the transducer refuses it). ``reason`` and
    ``source`` are mandatory: a classification with no citation is not admitted.
    """

    drug: str
    kind: str
    reason: str
    source: str

    _KINDS = ("silent_antagonist", "partial_agonist")

    def __post_init__(self):
        if not self.drug or self.drug != self.drug.strip().lower():
            raise ValueError("drug must be a non-empty lowercased name")
        if self.kind not in self._KINDS:
            raise ValueError(
                "kind must be one of %s; got %r" % (self._KINDS, self.kind)
            )
        if not self.reason.strip():
            raise ValueError("%s: reason must be recorded" % self.drug)
        if not self.source.strip():
            raise ValueError("%s: a source citation is required" % self.drug)

    @property
    def transducible(self):
        """True when occupancy maps to effect without an intrinsic-activity term."""
        return self.kind == "silent_antagonist"


_HALOPERIDOL = D2FunctionalClass(
    drug="haloperidol",
    kind="silent_antagonist",
    reason=(
        "Silent D2 antagonist with no measurable D2 agonist intrinsic activity: "
        "occupancy removes receptors from dopamine signalling, so occupancy is "
        "the fractional loss of D2 tone and the occupancy identity "
        "abs(effect) = occupancy is a sign-convention re-expression, not an "
        "invented measurement."
    ),
    source=(
        "[MailmanMurthy2010] Mailman & Murthy, Curr Pharm Des 16(5):488-501, "
        "PMID 19909227 -- groups haloperidol among antipsychotics with no D2 "
        "agonist intrinsic activity, in contrast to aripiprazole; the "
        "haloperidol dose-to-occupancy chain rests on the [Kapur1997] "
        "plasma-occupancy relationship (Am J Psychiatry 154:514-520, "
        "PMID 9201802)."
    ),
)

_ARIPIPRAZOLE = D2FunctionalClass(
    drug="aripiprazole",
    kind="partial_agonist",
    reason=(
        "D2 partial agonist whose intrinsic activity is irreducibly "
        "system-dependent, so no single Emax can be honestly assigned. "
        "[Burris2002] PMID 12065741 reports efficacy relative to dopamine "
        "varying from 25% (no spare receptors) to 90% (with receptor reserve) "
        "in one human-D2L/CHO cAMP assay; [MailmanMurthy2010] PMID 19909227 "
        "finds it spans the whole range from pure antagonist to full agonist "
        "across readouts. No in-vivo human D2 intrinsic-activity fraction has "
        "ever been measured (searched 2026-09-08, PubMed/Europe PMC; not "
        "found). The clinical signature a single antagonist transducer cannot "
        "reproduce: ~85-95% striatal D2 occupancy that is largely EPS-sparing "
        "([Mamo2007] PMID 17728427; [Yokoi2002] PMID 12093598; [Grunder2008] "
        "PMID 18381901)."
    ),
    source=(
        "[Burris2002] PMID 12065741; [MailmanMurthy2010] PMID 19909227; "
        "[Mamo2007] PMID 17728427; [Yokoi2002] PMID 12093598; "
        "[Grunder2008] PMID 18381901"
    ),
)

_D2_FUNCTIONAL_CLASSES = MappingProxyType(
    {_HALOPERIDOL.drug: _HALOPERIDOL, _ARIPIPRAZOLE.drug: _ARIPIPRAZOLE}
)


def registered_d2_functional_classes():
    """The drugs with a sourced D2 functional classification, sorted."""
    return tuple(sorted(_D2_FUNCTIONAL_CLASSES))


def d2_functional_class(drug):
    """The sourced :class:`D2FunctionalClass` for ``drug``.

    Raises :class:`KeyError` naming the classified drugs when ``drug`` has none,
    rather than guessing. Classification is not read off a binding table: it
    requires a functional assay, which is what the cited sources supply.
    """
    key = str(drug).strip().lower()
    try:
        return _D2_FUNCTIONAL_CLASSES[key]
    except KeyError:
        raise KeyError(
            "no D2 functional classification for %r; classified drugs are %s"
            % (drug, ", ".join(registered_d2_functional_classes()))
        )


def d2_effect_from_occupancy(occupancy, baseline_occupancy=0.0, emax=1.0):
    """The raw transducer: caller supplies Emax and takes responsibility for it.

    This is the drug-agnostic construction. It does not know whether the
    occupancy it is handed came from an antagonist or a partial agonist, so it
    applies whatever ``emax`` the caller passes. Use :func:`d2_effect_for_drug`
    when the drug is known and a partial agonist must be refused.
    """
    return D2Effect(occupancy=float(occupancy), baseline_occupancy=float(baseline_occupancy), emax=float(emax))


def d2_effect_for_drug(occupancy, drug, baseline_occupancy=0.0):
    """Transduce occupancy to effect for a named drug, refusing partial agonists.

    A silent D2 antagonist is transduced at ``emax = 1`` -- the occupancy
    identity, honest for an antagonist. A D2 partial agonist raises
    :class:`PartialAgonistTransductionError`, because its occupancy overstates
    effect by an intrinsic activity that has no single sourced value. An
    unclassified drug raises :class:`KeyError` rather than defaulting to the
    identity.
    """
    functional_class = d2_functional_class(drug)
    if not functional_class.transducible:
        raise PartialAgonistTransductionError(
            "%s is a D2 %s; the occupancy-limited transducer will not act on it. "
            "Occupancy overstates effect by an intrinsic activity with no single "
            "sourced value, so emax=1 (abs(effect)=occupancy) is not honest here. "
            "%s"
            % (functional_class.drug, functional_class.kind, functional_class.reason)
        )
    return d2_effect_from_occupancy(
        occupancy, baseline_occupancy=baseline_occupancy, emax=1.0
    )
