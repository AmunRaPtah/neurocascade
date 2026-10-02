"""Leaf primitives for the species gate and P29.1 translation inputs.

This module holds the parts of the species boundary that carry NO dependency on
the fingerprint / circuit / connectome layers, so a module on the v1 prediction
path (``human.dose_to_occupancy``, ``human.kp_uu``) can import a `TranslatedInput`
type or ask whether a translation tag admits a number without dragging the eager
package tree -- and the frozen layers it holds -- into its import closure
(``tests/test_v1_frozen_layers.py``). ``species_gate`` re-imports everything here,
so its public API (``species_gate.normalize_species``,
``species_gate.UnsourcedParameterError``, ``species_gate.TranslationTag`` ...) is
unchanged; ``species_gate`` adds the binding/PK/anchor machinery that genuinely
needs the fingerprint layer on top.
"""

import re


class UnsourcedParameterError(LookupError):
    """Raised when a parameter whose provenance does not license the use it
    is being put to reaches a prediction or scenario builder.

    The canonical definition lives HERE since P11.2, because the species
    gate raises it across four layers and the gate module may not import
    `legacy.drug_pk` (nothing outside `legacy/` may). `legacy.drug_pk`
    re-imports it, so every pre-existing import path and except-clause keeps
    working, and a rodent PK lookup and a species-gate refusal are caught by
    the same clause -- both are "this parameter does not exist for this use".

    Two distinct situations raise it, and the messages distinguish them:

    * the parameter has been searched for and not found (drug_pk's original
      use -- see `SEARCHED_AND_EMPTY`);
    * the parameter EXISTS but was measured in a non-human species, so it
      may not enter a human prediction (the species gate's use).
    """


# ---------------------------------------------------------------------------
# Species tokens
# ---------------------------------------------------------------------------

#: The canonical token for the modelling species.
HUMAN = "human"

#: Recognized spellings -> canonical token. Deliberately SHORT: a name is
#: mapped only when it names the animal unambiguously. Cell-line names
#: ("HEK293", "CHO"), preparation words ("CLONED") and strain names are NOT
#: here -- they do not say which species' receptor was measured, which is
#: exactly the fact the gate exists to check, so they stay unrecognized and
#: refuse.
SPECIES_SYNONYMS = {
    "human": (
        "human", "homo sapiens", "homo sapien", "human receptor",
        "humans",
    ),
    "mouse": ("mouse", "mus musculus", "mice"),
    "rat": ("rat", "rattus norvegicus", "rats"),
    "rhesus": ("rhesus", "rhesus macaque", "rhesus monkey",
               "macaca mulatta"),
    "baboon": ("baboon",),
}

_REVERSED_SPECIES = {
    spelling: canonical
    for canonical, spellings in SPECIES_SYNONYMS.items()
    for spelling in spellings
}


def normalize_species(value):
    """The canonical species token for `value`, or None when there is none.

    None means NO CLAIM was made -- the argument was None or empty -- which
    the gate treats as "no provenance to check" and lets through. Anything
    else that does not map is returned lowercased and stripped, so it can
    never compare equal to `HUMAN` and will surface verbatim in the refusal.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    mapped = _REVERSED_SPECIES.get(text)
    if mapped is not None:
        return mapped
    # Tolerate decorations around a recognized name: "rat (probable)",
    # "human recombinant", "Mus musculus (C57BL/6J)".
    for spelling, canonical in _REVERSED_SPECIES.items():
        if re.search(r"\b%s\b" % re.escape(spelling), text):
            return canonical
    return text




# ---------------------------------------------------------------------------
# Declared translation inputs (P29.1)
# ---------------------------------------------------------------------------
#
# P11.2 refused every non-human number at a driving position. Phase 29 needs
# one narrow exception: preclinical-to-human translation, where a rat Kp,uu or
# a rat-hepatocyte intrinsic clearance is the INPUT to a published method whose
# whole purpose is to carry it to human. The exception is a tag, never a
# species: the same rat number with no tag is still refused, and a tag names
# both the method (each entry cited to the paper that defines it) and the
# measurement it was applied to. What the gate hands back is a `TranslatedInput`,
# which the prediction must carry in its provenance so a reader can see which of
# its inputs crossed a species and by what method.
#
# A translation input is not an anchor. An anchor is a matched pair kept to
# measure transfer error and never drives anything; a translation input drives
# a prediction and declares that it does.


class TranslationMethod(object):
    """A published method for carrying a non-human measurement to human."""

    __slots__ = ("key", "name", "citation")

    def __init__(self, key, name, citation):
        if not citation:
            raise ValueError("a translation method needs a citation")
        self.key = str(key)
        self.name = str(name)
        self.citation = str(citation)

    def __repr__(self):
        return "TranslationMethod(%r, %s)" % (self.key, self.citation)


#: The methods a tag may name. Citations verified against the Europe PMC
#: records on 2026-10-02. Each is the paper that defines or first validated the
#: method as a species bridge; none of them supplies a parameter value.
TRANSLATION_METHODS = {
    method.key: method
    for method in (
        TranslationMethod(
            "allometric-scaling",
            "simple allometry of a clearance or volume across species by "
            "body weight, Y = a * W**b",
            "Boxenbaum 1982, J Pharmacokinet Biopharm 10:201-227, "
            "PMID 7120049",
        ),
        TranslationMethod(
            "ivive",
            "in-vitro-in-vivo extrapolation of hepatic clearance from "
            "microsomal intrinsic clearance (in-vitro half-life approach, "
            "with microsomal and blood binding)",
            "Obach 1999, Drug Metab Dispos 27:1350-1359, PMID 10534321",
        ),
        TranslationMethod(
            "kp-uu-transfer",
            "rat unbound brain:plasma ratio (Kp,uu,brain) carried to human "
            "unbound brain exposure",
            "Friden 2009, J Med Chem 52:6233-6243, PMID 19764786 (rank order "
            "of rat and human Kp,uu,CSF similar across 43 drugs)",
        ),
    )
}


class TranslationTag(object):
    """The declaration that lets one non-human measurement into a human
    prediction: which registered method carries it, from which species, and
    the citation of the measurement itself."""

    __slots__ = ("method", "source_species", "source", "detail")

    def __init__(self, method, source_species, source, detail=""):
        if method not in TRANSLATION_METHODS:
            raise ValueError(
                "translation method %r is not registered; known methods: %s"
                % (method, ", ".join(sorted(TRANSLATION_METHODS)))
            )
        species = normalize_species(source_species)
        if species is None or species == HUMAN:
            raise ValueError(
                "a translation tag declares a crossing FROM a non-human "
                "species; got %r" % (source_species,)
            )
        if species not in SPECIES_SYNONYMS:
            raise ValueError(
                "translation source species %r is not a recognized species "
                "token, so the crossing it declares cannot be checked"
                % (source_species,)
            )
        if not str(source).strip():
            raise ValueError(
                "a translation tag must cite the measurement it translates"
            )
        self.method = TRANSLATION_METHODS[method]
        self.source_species = species
        self.source = str(source)
        self.detail = str(detail)

    def __repr__(self):
        return "TranslationTag(%r, from %r, %s)" % (
            self.method.key, self.source_species, self.source
        )


class TranslatedInput(object):
    """What the gate returns for an admitted translation input: the
    parameter's name and layer, its species, and the tag that admitted it.
    This is the provenance record a prediction carries."""

    __slots__ = ("layer", "label", "species", "source", "tag")

    def __init__(self, layer, label, species, source, tag):
        self.layer = str(layer)
        self.label = str(label)
        self.species = species
        self.source = str(source) or tag.source
        self.tag = tag

    @property
    def method(self):
        return self.tag.method.key

    def describe(self):
        return "%s: %s translated from %s by %s [%s; method: %s]" % (
            self.layer, self.label, self.species, self.tag.method.key,
            self.source, self.tag.method.citation,
        )

    def __repr__(self):
        return "TranslatedInput(%r, %s via %s)" % (
            self.label, self.species, self.method
        )


def _admit_translation(normalized, layer, label, source, translation):
    if not isinstance(translation, TranslationTag):
        raise TypeError(
            "translation must be a species_gate.TranslationTag; got %r"
            % (translation,)
        )
    if normalized is None or normalized == HUMAN:
        raise ValueError(
            "%s: %s carries provenance species %r, so there is no species "
            "crossing for a translation tag to declare" % (layer, label,
                                                           normalized)
        )
    if normalized != translation.source_species:
        raise ValueError(
            "%s: %s was measured in %r but its translation tag declares a "
            "%r source; a tag licenses only the species it names"
            % (layer, label, normalized, translation.source_species)
        )
    return TranslatedInput(layer=layer, label=label, species=normalized,
                           source=source, tag=translation)



def check_translation(species, *, layer, label, source="", translation=None):
    """The translation-aware core of the species gate, free of fingerprint deps.

    Returns None for a human or no-claim parameter, a `TranslatedInput` for a
    non-human parameter carrying a valid `TranslationTag`, and raises
    `UnsourcedParameterError` for a non-human parameter with no tag -- the same
    refusal `species_gate.gate` gives, so a caller on the v1 path gets the P29.1
    rule without importing the full gate. ``species_gate.gate`` delegates its
    translation handling here so there is one implementation of the rule.
    """
    normalized = normalize_species(species)
    if translation is not None:
        return _admit_translation(normalized, layer, label, source, translation)
    if normalized is None or normalized == HUMAN:
        return None
    raise UnsourcedParameterError(
        "%s: %s carries provenance species %r, which is not %r, so it cannot "
        "enter a human prediction. %sIt is not a declared translation input: a "
        "non-human number crosses the species boundary only under a "
        "translation.TranslationTag naming a cited method."
        % (layer, label, normalized, HUMAN,
           "Source of the parameter in hand: %s. " % source if source else "")
    )


__all__ = [
    "UnsourcedParameterError",
    "HUMAN",
    "SPECIES_SYNONYMS",
    "normalize_species",
    "TranslationMethod",
    "TRANSLATION_METHODS",
    "TranslationTag",
    "TranslatedInput",
    "check_translation",
]
