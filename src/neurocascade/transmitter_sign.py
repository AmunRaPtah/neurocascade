"""Transmitter classes and the sign each contributes to a synaptic drive term.

This is the species-neutral core of the signing rule. Which transmitter a given
region's projection neurons release is a species-specific anatomical fact and
lives with that species' connectome (`legacy/connectome/neurotransmitter.py`
for the mouse CCF, `human/neurotransmitter.py` for the human parcellation). What
a transmitter class *does* to a Wilson-Cowan drive term does not depend on the
species, so it lives here and both connectomes import it.

Hoisted out of `connectome/neurotransmitter.py` in P11.1. Before the hoist the
human layer imported these constants from the mouse package, which made the
species quarantine unenforceable: `legacy/` could not be separated from a human
prediction while a human module depended on it. Neither species depends on the
other now.

The rule, restated from the module this came out of:

    sign(edge j -> i) = sign of the transmitter released by the
                        LONG-RANGE PROJECTION neurons of source region j,

        glutamatergic  -> +1
        GABAergic      -> -1
        anything else  -> no sign

`None` means "this class has no sign in a Wilson-Cowan connectivity matrix",
which is a statement about the model rather than an admission of ignorance. A
neuromodulatory projection does not add a signed fast-synaptic drive term; it
changes how a target responds to the drive it already gets, which is what the
receptor and projection layers represent.
"""

# --- transmitter classes ---------------------------------------------------

GLUTAMATERGIC = "glutamatergic"
GABAERGIC = "gabaergic"
DOPAMINERGIC = "dopaminergic"
SEROTONERGIC = "serotonergic"
NORADRENERGIC = "noradrenergic"
CHOLINERGIC = "cholinergic"
MIXED = "mixed"

#: Sign a transmitter class contributes to a fast-synaptic drive term.
TRANSMITTER_SIGN = {
    GLUTAMATERGIC: 1.0,
    GABAERGIC: -1.0,
    DOPAMINERGIC: None,
    SEROTONERGIC: None,
    NORADRENERGIC: None,
    CHOLINERGIC: None,
    MIXED: None,
}

__all__ = [
    "CHOLINERGIC",
    "DOPAMINERGIC",
    "GABAERGIC",
    "GLUTAMATERGIC",
    "MIXED",
    "NORADRENERGIC",
    "SEROTONERGIC",
    "TRANSMITTER_SIGN",
]
