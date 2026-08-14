"""P8.1: an arbitrary compound's BINDING FINGERPRINT as model input.

What this module is for
=======================

Every receptor system in this codebase before P8.1 is a *scenario*: a named
drug whose constants were sourced by hand, wired into a named circuit, in a
module written for it. That is the right shape for calibration and the wrong
shape for prediction, because a prediction tool has to accept a compound it
has never seen, described the way binding data is actually published:

    a table of Ki values, one row per (compound, target), plus -- separately,
    because no binding database publishes it -- an efficacy per target.

This module is the interface that takes that table and builds the compound's
ligand set across every modelled receptor and transporter, so that

    predict(fingerprint, dose, route, species)

is a call a caller can make about a compound this repository contains no
module for. P8.2 is what turns the receptor-layer answer this returns into a
behavioural profile; P8.1 is the input side and stops where the evidence
does (see "Where this stops", below).

The PDSP row shape, checked rather than assumed
===============================================

The NIMH Psychoactive Drug Screening Program Ki database
(https://pdspdb.unc.edu/kidb2/kidb/web/) publishes one row per determination.
Its result table's columns, read off the live database on 2026-08-14, are::

    Ki ID | Receptors | Sources | Species | Hot Ligands | Test Ligands
          | Ki Value | Citations | Submitted At | Updated At

with, for example::

    100956 | NET | CELLS | HEK293 | 3H-CIT | MAZINDOL | 0.738 | Reith ME, ...

so `Receptors` carries a short target symbol, `Test Ligands` the compound,
`Hot Ligands` the radioligand the compound was competed against, `Sources`
the preparation and `Species` its origin (a cell line, for a cell-based
assay), and `Ki Value` is in nM. `PDSPRow` and
`BindingFingerprint.from_pdsp_rows` take exactly those fields.

**There is no efficacy column, and that is a property of the assay, not an
omission.** A competition binding experiment measures how tightly a compound
occupies a site. What the occupied receptor then *does* -- full agonism,
partial agonism, neutral antagonism, inverse agonism -- is a functional
measurement (GTP-gamma-S, cAMP, electrophysiology) that the same paper may or
may not report. So this module REQUIRES an efficacy per retained target and
raises `MissingEfficacyError` naming the targets it was not given one for.
It does not default efficacy to 1.0, and it does not infer it from affinity,
from the compound's name, or from what other ligands at that target do.
Defaulting it would be the cheapest possible way to invent a number, and it
would be invisible in the output.

Ki -> kon/koff, and the assumption that carries
===============================================

The conversion is `receptor.ligand_from_ki`, which states the two
assumptions in full. In one sentence:

    THE Ki FIXES koff/kon AND NEITHER RATE, SO SPLITTING IT REQUIRES ONE MORE
    NUMBER; THIS MODULE SUPPLIES `kon` PER TARGET AND SETS `koff = kon * Ki`,
    WHICH REPRODUCES THE MEASURED AFFINITY EXACTLY AND PUTS THE ENTIRE
    BINDING TIMESCALE ON THE SUPPLIED `kon`.

Where each target's `kon` comes from is not uniform across this codebase and
must not be flattened, so `TargetSpec.kon_provenance` records which of four
standings applies, and `target_report()` prints the whole table:

``measured-at-site``
    A radioligand's association rate measured at THIS receptor. The strongest
    case: 5-HT1A, 5-HT2A, alpha1, alpha2, beta, H1, A2A, D1, D2, delta and
    kappa opioid.
``borrowed-from-subtype``
    Measured for one ligand at a NEIGHBOURING subtype of the same receptor
    family. One target: mu opioid, which takes diprenorphine's kappa `kon`
    exactly as `opioid._ligand` already does for every mu ligand it builds.
``assumed``
    Not measured anywhere for this site, and a value chosen to set a
    timescale. One target: the GABA-A benzodiazepine site, whose `kon_shared`
    `scenarios.py` already flags as "assumed, not measured".
``illustrative``
    `ILLUSTRATIVE_KON = 0.01`, the shared placeholder that `acetylcholine.py`,
    `glutamate.py` and `cannabinoid.py` each defined for the sites where no
    binding kinetics exist at all: alpha4beta2, alpha7, M1, M2, NMDA, AMPA,
    CB1.
``not-needed``
    The three monoamine transporters. `transport.py` consumes a Ki DIRECTLY,
    as the competitive term `1 + sum(C_i/Ki_i)` on the apparent Km, and
    never forms a rate pair -- so a fingerprint entry at DAT, SERT or NET
    carries NO koff assumption whatsoever. This is the one part of the
    fingerprint path that is assumption-free downstream of `Kd = Ki`, and it
    is worth knowing which part that is.

A fingerprint can ADD a ligand; it can never REDEFINE a system
==============================================================

Every entry in `TARGETS` reaches its receptor through a zero-argument
callable returning that module's OWN `engine.ReceptorSystem` or
`engine.TransporterSystem`. This is `projection_scenarios.ReceptorPlacement`'s
rule, applied to a second problem for the same reason: a port "cannot quietly
redefine a receptor: it can only move it". Here a fingerprint cannot quietly
redefine one either -- it can only join the competitive pool. The coupling,
the connection modulation, the endogenous ligands, the desensitization rates
and the PK models of everything already at the site come through untouched,
and `add_to_system` refuses by default to overwrite a ligand that is already
there.

Which pool a fingerprint ligand joins is the caller's declaration
=================================================================

`Kd = Ki` reads a competition constant as an orthosteric dissociation
constant, which is what it is for a ligand competing with the radioligand at
the labelled site, and is not what it is for an open-channel blocker or an
allosteric modulator. A Ki table does not say which a compound is. So
`FingerprintEntry.pool` defaults to `"orthosteric"` -- the reading the datum
supports -- and a caller who knows better says `pool="channel"`, which is
refused at any site that has no channel-blocker pool rather than silently
demoted into the competitive one.

Four targets have such a pool -- NMDA, AMPA and BOTH nicotinic sites, since
`DesensitizingReceptorSite` carries channel blockers without subclassing
`ChannelBlockingReceptorSite` -- and `_CHANNEL_KON` gives each its own `kon`
and standing. NMDA's is the one place a fingerprint ligand gets a MEASURED
rate ([Dravid2007]'s MK-801 association rate), so a channel entry there is
better founded than an orthosteric one; the other three keep their site's
illustrative value and say so.

Where this stops, and why
=========================

`predict()` returns the RECEPTOR LAYER: per target, the compound's fractional
occupancy and the site's activation over time, from the administered dose
through the Phase 1 PK bridge and the Phase 1.5 brain partition. That is
exactly the part of the chain a binding fingerprint plus a PK model
determines, with no unfitted parameter anywhere in it.

It deliberately stops there. Turning an activation into a behavioural score
needs, per target, a `mechanism_gain` and a region-wise density
(`coupling[region] = mechanism_gain * rho[region]`, P4.2/P7.3), and exactly
ONE such gain has ever been fitted in this repository. Composing every
target's activation onto one whole-brain network and reporting a behavioural
profile is P8.2, and doing it here with unfitted gains would produce a
plausible-looking number whose magnitude nothing supports. What P8.1 hands
P8.2 is a wired `engine.ReceptorSystem` per target
(`receptor_systems_for`), which is the object a whole-brain composition
needs.

Two further limits, stated rather than discovered later:

* **No endogenous tone.** The other ligands at each site keep their own PK
  models and are dosed at zero, so a `predict()` run is a drug arriving at an
  empty pool. Tone is P6.2's `endogenous_tone` and Phase 6's projection
  systems, which need a circuit; `predict` has none by design.
* **Species is the PK's, not the binding's.** `species` selects the
  drug/species PK parameter set and the brain partition. The binding
  constants of the OTHER ligands at each site are whatever their own module
  sourced, and those species differ between modules and are documented
  there. A fingerprint's own rows carry the species the Ki was measured in
  (`FingerprintEntry.species`), which is recorded and reported and is not
  checked against the PK species, because a compound's affinity and its
  disposition are routinely measured in different preparations and refusing
  that combination would refuse most of the literature.
"""

import numpy as np
from scipy.integrate import solve_ivp

from . import acetylcholine as _ach
from . import adenosine as _ado
from . import cannabinoid as _cb
from . import dopamine as _da
from . import drug_pk as _drug_pk
from . import glutamate as _glu
from . import histamine as _ha
from . import monoamine_transporters as _mat
from . import noradrenaline as _na
from . import opioid as _op
from . import scenarios as _scenarios
from . import serotonin as _5ht
from .engine import ReceptorSystem, TransporterSystem
from .pk import as_ode_pk_model
from .receptor import (
    ChannelBlockingReceptorSite,
    DesensitizingReceptorSite,
    ReceptorSite,
    ligand_from_ki,
)
from .transport import MonoamineTransporter, TransporterLigand


class UnknownTargetError(LookupError):
    """A fingerprint names a target this codebase does not model.

    Raised only where the caller asked for that target by name. Building a
    fingerprint from a PDSP table does NOT raise: an arbitrary compound's
    published panel routinely covers targets outside any one model, and those
    rows are recorded in `BindingFingerprint.unmodelled` rather than thrown
    away or guessed at.
    """


class MissingEfficacyError(ValueError):
    """A modelled target has a Ki and no efficacy. See the module docstring."""


#: The four standings a target's `kon` can have, plus the transporter case
#: where no `kon` is formed at all. See the module docstring.
KON_PROVENANCES = (
    "measured-at-site",
    "borrowed-from-subtype",
    "assumed",
    "illustrative",
    "not-needed",
)

#: The pools a fingerprint ligand may be asked to join.
POOLS = ("orthosteric", "channel")


def normalize_target(name):
    """Fold a target name to its lookup key: lowercase, alphanumerics only.

    So `5-HT2A`, `5HT2A`, `5 HT 2a` and `5ht2a` are one key, and `Alpha-2` and
    `alpha2` are one key. This is a normalization THIS PROJECT defines over
    the names it accepts; it is not something PDSP publishes, and it is
    deliberately blunt: it cannot merge two targets that differ by more than
    punctuation and case, so it can widen what is accepted without ever
    silently mapping one receptor onto another.
    """
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


class TargetSpec(object):
    """One modelled target: how to reach it, and what its `kon` is worth.

    Parameters
    ----------
    name : str
        Canonical name, the key of `TARGETS` and of every result this module
        returns.
    kind : "receptor" or "transporter"
    system : callable
        Zero-argument, returning that module's own `engine.ReceptorSystem`
        or `engine.TransporterSystem`. This module only ever READS it; see
        the module docstring.
    kon : float or None
        nM^-1 min^-1, used as `koff = kon * Ki`. `None` for a transporter,
        which consumes the Ki directly and forms no rate pair.
    kon_provenance : one of `KON_PROVENANCES`
    kon_source : str
        Where the number is from, in words, including its citation key where
        the module has one.
    aliases : sequence of str
        Extra spellings accepted for this target, normalized by
        `normalize_target`. PDSP-style symbols and HGNC gene symbols both.
    note : str
        Anything a caller needs to know before reading a number off this
        target: the site's own species, an unusual `efficacy` meaning, a
        pool that is not a plain competitive one.
    gene : str or None
        The mouse gene symbol in `expression.genes.RECEPTOR_GENES` whose ISH
        map stands for this target's regional density (P4.1/P4.2). THIS
        REGISTRY IS THE FIRST PLACE THE TWO NAME SPACES ARE JOINED: before
        P8.1 the pharmacological names (`opioid.RECEPTORS`,
        `endogenous_tone.TONE_TARGETS`) and the anatomical ones
        (`RECEPTOR_GENES`, keyed by gene symbol) never met, and P8.2 needs
        them joined because a whole-brain prediction is
        `coupling[region] = mechanism_gain * rho[region]` with `rho` read off
        the gene's map.
    gene_note : str
        Required wherever the receptor is a MULTI-SUBUNIT assembly and one
        gene therefore cannot be the receptor. `genes.py` already says this
        about GABA-A ("a pentamer") and it is equally true of the nicotinic,
        NMDA and AMPA entries: for those the gene is a stated proxy, not an
        identity. The requirement cannot be checked in this constructor,
        which has no way to know whether a receptor is an assembly, so it is
        checked in `test_fingerprint.py::test_the_multi_subunit_targets_are_
        exactly_the_ones_flagged_as_proxies`, which pins the flagged set
        exactly rather than only requiring it to be non-empty.
    """

    def __init__(self, name, kind, system, kon, kon_provenance, kon_source,
                 aliases=(), note="", gene=None, gene_note=""):
        if kind not in ("receptor", "transporter"):
            raise ValueError("kind must be 'receptor' or 'transporter'")
        if kon_provenance not in KON_PROVENANCES:
            raise ValueError(
                "kon_provenance must be one of %s; got %r"
                % (", ".join(KON_PROVENANCES), kon_provenance)
            )
        if (kon is None) != (kon_provenance == "not-needed"):
            raise ValueError(
                "a target has a kon exactly when its provenance is not "
                "'not-needed'; %r has kon=%r and provenance %r"
                % (name, kon, kon_provenance)
            )
        if not gene:
            raise ValueError(
                "%r needs the gene whose density map stands for it; P8.2 "
                "reads one off every target" % name
            )
        self.name = name
        self.kind = kind
        self.system = system
        self.kon = None if kon is None else float(kon)
        self.kon_provenance = kon_provenance
        self.kon_source = kon_source
        self.aliases = tuple(aliases)
        self.note = note
        self.gene = gene
        self.gene_note = gene_note

    @property
    def is_transporter(self):
        return self.kind == "transporter"

    def __repr__(self):
        return "TargetSpec(%r, %s, kon=%r [%s])" % (
            self.name, self.kind, self.kon, self.kon_provenance
        )


def _kon_of(ligands, ligand_name):
    """One named ligand's `kon`, read off the module's own ligand list.

    Used where a module keeps its site `kon` inside a calibrated ligand
    rather than exporting it as a constant (D2, the benzodiazepine site), so
    that this registry cannot drift from the value the scenario actually
    uses. `test_fingerprint.py` re-derives every entry the same way.
    """
    for ligand in ligands:
        if ligand.name == ligand_name:
            return float(ligand.kon)
    raise LookupError("no ligand named %r" % (ligand_name,))


#: Every modelled receptor and transporter target, keyed by canonical name.
#:
#: The `kon` column is the whole point of the table and is assembled from
#: each module's own constants rather than restated, so a module that changes
#: its site `kon` changes this too. `test_fingerprint.py` checks each value
#: against the module it came from, and checks that the set of targets equals
#: what the receptor modules actually build.
TARGETS = {}


def _register(spec):
    for key in (spec.name,) + spec.aliases:
        normalized = normalize_target(key)
        clash = _ALIAS_INDEX.get(normalized)
        if clash is not None and clash != spec.name:
            raise ValueError(
                "alias %r maps to both %r and %r" % (key, clash, spec.name)
            )
        _ALIAS_INDEX[normalized] = spec.name
    TARGETS[spec.name] = spec
    return spec


_ALIAS_INDEX = {}

# --- opioid receptors ------------------------------------------------------

_register(TargetSpec(
    name="MOR", kind="receptor", system=_op.mu_receptor_system_calibrated,
    kon=_op.SITE_KON.get("mu", _op.KAPPA_KON_DIPRENORPHINE),
    kon_provenance="borrowed-from-subtype",
    kon_source=(
        "diprenorphine's measured association rate at KAPPA "
        "(opioid.KAPPA_KON_DIPRENORPHINE), used at mu because no mu "
        "association rate is available. This is not a new decision: "
        "opioid._ligand already falls back to exactly this value for every "
        "mu ligand it builds, and the fallback is reused here rather than "
        "replaced so that a fingerprint ligand and a hand-built one at mu "
        "share one timescale."
    ),
    aliases=("mu", "mu-opioid", "OPRM1", "MU OPIOID", "u-opioid"),
    note=(
        "The site is scenarios.mu_opioid_receptor_site_calibrated (Track A: "
        "fentanyl [Mann2022]/[Volpe2011], naloxone [Mann2022]). Those two "
        "ligands were built by DERIVATION 2 -- their kon comes from a "
        "measured koff and a measured Ki -- so a fingerprint ligand at mu "
        "reproduces their affinities and not their rates. See "
        "docs/architecture.md, P8.1."
    ),
    gene='Oprm1',
))

_register(TargetSpec(
    name="DOR", kind="receptor", system=_op.receptor_system_delta,
    kon=_op.DELTA_KON_PCL_DPDPE, kon_provenance="measured-at-site",
    kon_source="pCl-DPDPE at the delta site (opioid.DELTA_KON_PCL_DPDPE)",
    aliases=("delta", "delta-opioid", "OPRD1", "DELTA OPIOID"),
    gene='Oprd1',
))

_register(TargetSpec(
    name="KOR", kind="receptor", system=_op.receptor_system_kappa,
    kon=_op.KAPPA_KON_DIPRENORPHINE, kon_provenance="measured-at-site",
    kon_source="diprenorphine at the kappa site (opioid.KAPPA_KON_DIPRENORPHINE)",
    aliases=("kappa", "kappa-opioid", "OPRK1", "KAPPA OPIOID"),
    note=(
        "The kappa system's action is a connection_modulation on the "
        "NAc<-VTA_DA projection, not a drive coupling, so its `coupling` is "
        "empty by construction."
    ),
    gene='Oprk1',
))

# --- dopamine --------------------------------------------------------------

_register(TargetSpec(
    name="D1", kind="receptor", system=_da.receptor_system_d1,
    kon=_da.D1_KON_DOPAMINE, kon_provenance="measured-at-site",
    kon_source="dopamine at D1 (dopamine.D1_KON_DOPAMINE)",
    aliases=("DRD1", "D1A"),
    gene='Drd1',
))

_register(TargetSpec(
    name="D2", kind="receptor", system=_da.receptor_system_d2,
    kon=_kon_of(_scenarios.d2_dopamine_ligands_calibrated(), "haloperidol"),
    kon_provenance="measured-at-site",
    kon_source=(
        "haloperidol at D2, 2.13e9 M^-1 min^-1 [Sykes2017], read off "
        "scenarios.d2_dopamine_ligands_calibrated(). The site's other "
        "measured kon is dopamine's 5.82 [Agren2021], 2.7-fold higher; the "
        "antagonist's is taken because a fingerprint's compound is more "
        "often a synthetic ligand than the endogenous transmitter, and the "
        "choice is recorded here because it is a choice."
    ),
    aliases=("DRD2", "D2L", "D2S"),
    gene='Drd2',
))

# --- serotonin -------------------------------------------------------------

_register(TargetSpec(
    name="5HT1A", kind="receptor", system=_5ht.receptor_system_5ht1a,
    kon=_5ht.HT1A_KON, kon_provenance="measured-at-site",
    kon_source="serotonin.HT1A_KON, the site's shared measured kon",
    aliases=("5-HT1A", "HTR1A", "5HT-1A"),
    gene='Htr1a',
))

_register(TargetSpec(
    name="5HT2A", kind="receptor", system=_5ht.receptor_system_5ht2a,
    kon=_5ht.HT2A_KON, kon_provenance="measured-at-site",
    kon_source="serotonin.HT2A_KON, the site's shared measured kon",
    aliases=("5-HT2A", "HTR2A", "5HT-2A"),
    note=(
        "serotonin.py carries two affinity states for this site "
        "(live-cell and antagonist-radioligand); the system builder's "
        "default, live-cell, is what a fingerprint ligand joins."
    ),
    gene='Htr2a',
))

# --- noradrenaline ---------------------------------------------------------

_register(TargetSpec(
    name="alpha1", kind="receptor", system=_na.receptor_system_alpha1,
    kon=_na.ALPHA1_KON_PRAZOSIN, kon_provenance="measured-at-site",
    kon_source="prazosin at alpha1 (noradrenaline.ALPHA1_KON_PRAZOSIN)",
    aliases=("a1", "alpha-1", "ADRA1A", "Alpha1A", "Alpha1B", "Alpha1D"),
    note=(
        "noradrenaline.py builds ONE alpha1 site from alpha1A constants and "
        "records 1B and 1D without instantiating them, so an alpha1B or "
        "alpha1D row is accepted into the alpha1 site and that substitution "
        "is what the alias asserts."
    ),
    gene='Adra1a',
))

_register(TargetSpec(
    name="alpha2", kind="receptor", system=_na.receptor_system_alpha2,
    kon=_na.ALPHA2_KON, kon_provenance="measured-at-site",
    kon_source="noradrenaline.ALPHA2_KON, the site's shared measured kon",
    aliases=("a2", "alpha-2", "ADRA2A", "Alpha2A", "Alpha2B", "Alpha2C"),
    gene='Adra2a',
))

_register(TargetSpec(
    name="beta", kind="receptor", system=_na.receptor_system_beta,
    kon=_na.BETA_KON_PROPRANOLOL, kon_provenance="measured-at-site",
    kon_source=(
        "propranolol at beta1, kinetically measured "
        "(noradrenaline.BETA_KON_PROPRANOLOL, [Sykes2022]). The same paper's "
        "18-ligand panel spans 230-fold in kon, and noradrenaline.py's own "
        "chemotype argument is that a shared kon is defensible WITHIN a "
        "chemotype and false across the panel -- so this number sets the "
        "site's timescale and is not a claim about any particular "
        "fingerprint ligand's association rate."
    ),
    aliases=("beta1", "beta2", "b1", "ADRB1", "Beta1", "Beta2", "Beta3"),
    gene='Adrb1',
))

# --- acetylcholine ---------------------------------------------------------

_register(TargetSpec(
    name="alpha4beta2", kind="receptor",
    system=_ach.receptor_system_alpha4beta2,
    kon=_ach.ILLUSTRATIVE_KON, kon_provenance="illustrative",
    kon_source="acetylcholine.ILLUSTRATIVE_KON; no binding kinetics exist for this site",
    aliases=("a4b2", "alpha4beta2 nicotinic", "CHRNA4", "nAChR alpha4beta2"),
    note=(
        "A DesensitizingReceptorSite. A Ki table says nothing about whether "
        "a compound drives desensitization, so a fingerprint ligand is added "
        "with k_des = 0, i.e. it occupies the orthosteric pool and does not "
        "drive the conformational change -- which is what the module already "
        "does for its competitive antagonists. A caller with a measured "
        "desensitization rate passes k_des/k_rec explicitly."
    ),
    gene='Chrna4',
    gene_note=(
        "PROXY: the receptor is an alpha4/beta2 pentamer and Chrnb2 is not in RECEPTOR_GENES, so the alpha4 map stands for the assembled receptor."
    ),
))

_register(TargetSpec(
    name="alpha7", kind="receptor", system=_ach.receptor_system_alpha7,
    kon=_ach.ILLUSTRATIVE_KON, kon_provenance="illustrative",
    kon_source="acetylcholine.ILLUSTRATIVE_KON; no binding kinetics exist for this site",
    aliases=("a7", "CHRNA7", "nAChR alpha7"),
    note="A DesensitizingReceptorSite; see alpha4beta2's note.",
    gene='Chrna7',
    gene_note=(
        "PROXY: alpha7 is a homopentamer, so this is the closest a single gene gets to a receptor in this table, and it is still a subunit map."
    ),
))

_register(TargetSpec(
    name="M1", kind="receptor", system=_ach.receptor_system_m1,
    kon=_ach.ILLUSTRATIVE_KON, kon_provenance="illustrative",
    kon_source="acetylcholine.ILLUSTRATIVE_KON; no binding kinetics exist for this site",
    aliases=("CHRM1", "muscarinic M1"),
    gene='Chrm1',
))

_register(TargetSpec(
    name="M2", kind="receptor", system=_ach.receptor_system_m2,
    kon=_ach.ILLUSTRATIVE_KON, kon_provenance="illustrative",
    kon_source="acetylcholine.ILLUSTRATIVE_KON; no binding kinetics exist for this site",
    aliases=("CHRM2", "muscarinic M2"),
    gene='Chrm2',
))

# --- glutamate -------------------------------------------------------------

_register(TargetSpec(
    name="NMDA", kind="receptor", system=_glu.receptor_system_nmda,
    kon=_glu.ILLUSTRATIVE_KON, kon_provenance="illustrative",
    kon_source=(
        "glutamate.ILLUSTRATIVE_KON, for the ORTHOSTERIC pool. The site's "
        "channel-blocker pool has a measured pair ([Dravid2007] MK-801) and "
        "a fingerprint entry declared pool='channel' takes "
        "glutamate.NMDA_CHANNEL_KON instead."
    ),
    aliases=("NMDAR", "GRIN1", "GRIN2A", "NMDA receptor"),
    note=(
        "A GlutamatergicTransmissionSite: its activation is NEGATIVE, the "
        "fraction of an existing excitatory weight the drugs remove, so "
        "`efficacy` here is an orthosteric agonist weight and an antagonist "
        "(efficacy 0) is what produces block. See glutamate.py."
    ),
    gene='Grin1',
    gene_note=(
        "PROXY: GluN1 is obligatory in every NMDA receptor, so its map bounds where the receptor can be; which GluN2 sits beside it, and therefore the subtype pharmacology glutamate.NMDA_SUBTYPE_IC50_MK801 records, it does not say."
    ),
))

_register(TargetSpec(
    name="AMPA", kind="receptor", system=_glu.receptor_system_ampa,
    kon=_glu.ILLUSTRATIVE_KON, kon_provenance="illustrative",
    kon_source="glutamate.ILLUSTRATIVE_KON; no binding kinetics for this site",
    aliases=("AMPAR", "GRIA1", "GRIA2", "AMPA receptor"),
    note="A GlutamatergicTransmissionSite; see NMDA's note.",
    gene='Gria2',
    gene_note=(
        "PROXY: GluA2 is present in most native AMPA receptors and is what sets their calcium impermeability, but an AMPA receptor is a tetramer of GluA1-4 and this is one of the four maps."
    ),
))

# --- histamine, cannabinoid, adenosine -------------------------------------

_register(TargetSpec(
    name="H1", kind="receptor", system=_ha.receptor_system_h1,
    kon=_ha.H1_KON_MEPYRAMINE, kon_provenance="measured-at-site",
    kon_source="mepyramine at H1 (histamine.H1_KON_MEPYRAMINE)",
    aliases=("HRH1", "histamine H1"),
    note=(
        "histamine.py also carries a seven-ligand measured kon panel "
        "(H1_KON_PANEL) spanning 330-fold, which is the error bar this "
        "single shared value carries and is the only such panel in the "
        "codebase."
    ),
    gene='Hrh1',
))

_register(TargetSpec(
    name="CB1", kind="receptor", system=_cb.receptor_system_cb1,
    kon=_cb.ILLUSTRATIVE_KON, kon_provenance="illustrative",
    kon_source="cannabinoid.ILLUSTRATIVE_KON; no CB1 binding kinetics were found",
    aliases=("CNR1", "cannabinoid CB1"),
    note=(
        "CB1 acts by connection_modulation with a MEASURED gain, so its "
        "`coupling` is empty and its activation scales a GABAergic weight."
    ),
    gene='Cnr1',
))

_register(TargetSpec(
    name="A2A", kind="receptor", system=_ado.receptor_system_a2a,
    kon=_ado.A2A_KON_NECA, kon_provenance="measured-at-site",
    kon_source="NECA at A2A (adenosine.A2A_KON_NECA)",
    aliases=("ADORA2A", "adenosine A2A", "A2a"),
    gene='Adora2a',
))

# --- the GABA-A benzodiazepine site ----------------------------------------

_register(TargetSpec(
    name="GABA_A_BZD", kind="receptor",
    system=_scenarios.gaba_a_receptor_system_calibrated,
    kon=_kon_of(_scenarios.gaba_a_benzodiazepine_ligands_calibrated(), "diazepam"),
    kon_provenance="assumed",
    kon_source=(
        "scenarios.gaba_a_benzodiazepine_ligands_calibrated's `kon_shared`, "
        "which that function's own comment flags as 'assumed, not measured'. "
        "Read off the ligand rather than restated so the two cannot drift."
    ),
    aliases=(
        "GABA-A benzodiazepine site", "benzodiazepine site", "BZD",
        "BZP Rat Brain Site", "GABAA BZD", "GABRA1",
    ),
    note=(
        "`efficacy` at this site is REPURPOSED as potentiation strength, not "
        "channel activation, because the site acts through "
        "connection_modulation (a benzodiazepine gates nothing itself). A "
        "fingerprint's efficacy for this target therefore has to be a "
        "potentiation figure, and 1.0 means 'as strong as the diazepam the "
        "P2.2 gain was fitted with'."
    ),
    gene='Gabra1',
    gene_note=(
        "PROXY: the benzodiazepine site is an interface between an alpha and the gamma2 subunit of a pentamer. Gabra1 is the most abundant alpha subunit and genes.py already flags that no single gene is the GABA-A receptor; alpha4 and alpha6 assemblies carry NO benzodiazepine site at all, so this map over-counts."
    ),
))

# --- monoamine transporters ------------------------------------------------


def _transporter_system(name):
    def build():
        return _mat.transporter_system(name)

    build.__name__ = "transporter_system_%s" % name.lower()
    return build


for _transporter in _mat.TRANSPORTERS:
    _register(TargetSpec(
        name=_transporter, kind="transporter",
        system=_transporter_system(_transporter),
        kon=None, kon_provenance="not-needed",
        kon_source=(
            "transport.py consumes a Ki directly, as the competitive term "
            "1 + sum(C_i/Ki_i) on the apparent Km, and forms no rate pair. "
            "No koff assumption is made for this target."
        ),
        gene={"DAT": "Slc6a3", "SERT": "Slc6a4", "NET": "Slc6a2"}[_transporter],
        aliases={
            "DAT": ("SLC6A3", "dopamine transporter"),
            "SERT": ("SLC6A4", "5-HTT", "serotonin transporter"),
            "NET": ("SLC6A2", "noradrenaline transporter",
                    "norepinephrine transporter"),
        }[_transporter],
        note=(
            "A Ki table cannot say whether a compound is a substrate-type "
            "releaser or a pure uptake blocker -- that is a separate release "
            "assay -- so a fingerprint entry becomes a PURE BLOCKER unless "
            "the caller supplies release_ec50 and release_emax. "
            "`efficacy` is not consumed at a transporter."
        ),
    ))

del _transporter


def resolve_target(name):
    """The canonical target name for `name`, or raise `UnknownTargetError`."""
    canonical = _ALIAS_INDEX.get(normalize_target(name))
    if canonical is None:
        raise UnknownTargetError(
            "%r is not a target this model carries. Known targets (aliases "
            "accepted, punctuation and case ignored): %s"
            % (name, ", ".join(sorted(TARGETS)))
        )
    return canonical


def is_modelled(name):
    """True when `name` resolves to a target in `TARGETS`."""
    return normalize_target(name) in _ALIAS_INDEX


def _as_efficacy(value):
    """`efficacy=` accepts a bare number or a `(value, source)` pair."""
    if isinstance(value, (tuple, list)):
        return (float(value[0]), str(value[1]))
    return (float(value), "")


def _fold_aliases(mapping, label, convert):
    """Resolve a `{target-or-alias: value}` mapping onto canonical keys.

    Raises when two aliases of the SAME target carry DIFFERENT values, which
    is otherwise a silent last-one-wins: `{"5-HT2A": 0.0, "HTR2A": 1.0}` is a
    caller who believes they described two receptors, and collapsing it would
    turn an antagonist into a full agonist without a word.
    """
    folded, seen_as = {}, {}
    for key, value in dict(mapping or {}).items():
        target = resolve_target(key)
        converted = convert(value)
        if target in folded and folded[target] != converted:
            raise ValueError(
                "%s gives two different values for %s, named as %r and %r: "
                "%r and %r. Those are the same target."
                % (label, target, seen_as[target], key,
                   folded[target], converted)
            )
        folded[target] = converted
        seen_as[target] = key
    return folded


def _unanimous(values):
    """The single value every row agrees on, or None when they disagree."""
    distinct = set(values)
    return distinct.pop() if len(distinct) == 1 else None


class PDSPRow(object):
    """One published binding determination, in the PDSP result table's shape.

    Field names follow that table's columns (see the module docstring);
    `receptor` and `test_ligand` are the singular forms of `Receptors` and
    `Test Ligands`. Everything except `receptor`, `test_ligand` and `ki_nM`
    is provenance that this module carries through and never computes on.
    """

    def __init__(self, receptor, test_ligand, ki_nM, species=None, source=None,
                 hot_ligand=None, reference=None, ki_id=None):
        self.receptor = str(receptor)
        self.test_ligand = str(test_ligand)
        self.ki_nM = float(ki_nM)
        self.species = species
        self.source = source
        self.hot_ligand = hot_ligand
        self.reference = reference
        self.ki_id = ki_id

    def __repr__(self):
        return "PDSPRow(%r at %r, Ki=%g nM)" % (
            self.test_ligand, self.receptor, self.ki_nM
        )


class FingerprintEntry(object):
    """One compound at one modelled target: an affinity and an efficacy.

    `efficacy` is required and is never defaulted; see the module docstring.
    `pool` declares which of a site's pools the ligand joins and defaults to
    the reading a competition Ki supports. `k_des`/`k_rec` are the
    desensitization rates at a nicotinic site, `None` for "not stated", which
    is what a Ki table supports; see `add_to_system`. `release_ec50` and
    `release_emax` are the transporter-only substrate parameters, absent by
    default because a Ki table does not contain them either, so a fingerprint
    entry at a transporter is a pure uptake blocker until a caller says
    otherwise.

    `species`, `source`, `hot_ligand` and `reference` are the row's own
    provenance, carried through and never computed on. Each is `None` when
    several determinations were merged and they disagreed about it, with the
    values listed in `note`: a Ki averaged over three preparations belongs to
    none of them.
    """

    def __init__(self, target, ki_nM, efficacy, efficacy_source="",
                 species=None, source=None, hot_ligand=None, reference=None,
                 pool="orthosteric", k_des=None, k_rec=None,
                 release_ec50=None, release_emax=0.0, note=""):
        if pool not in POOLS:
            raise ValueError("pool must be one of %s; got %r" % (POOLS, pool))
        self.target = target
        self.ki_nM = float(ki_nM)
        self.efficacy = float(efficacy)
        self.efficacy_source = efficacy_source
        self.species = species
        self.source = source
        self.hot_ligand = hot_ligand
        self.reference = reference
        self.pool = pool
        self.k_des = None if k_des is None else float(k_des)
        self.k_rec = None if k_rec is None else float(k_rec)
        self.release_ec50 = None if release_ec50 is None else float(release_ec50)
        self.release_emax = float(release_emax)
        self.note = note

    def __repr__(self):
        return "FingerprintEntry(%r, Ki=%g nM, efficacy=%g)" % (
            self.target, self.ki_nM, self.efficacy
        )


class BindingFingerprint(object):
    """A compound's affinities across this model's targets, plus efficacies.

    `entries` maps a CANONICAL target name to a `FingerprintEntry`.
    `unmodelled` maps a raw target name from the source table to the reason
    it is not in `entries`, so a panel that covers targets outside this model
    reports them instead of dropping them silently.
    """

    def __init__(self, compound, entries, unmodelled=None):
        self.compound = str(compound)
        self.entries = dict(entries)
        for name, entry in self.entries.items():
            if entry.target != name:
                raise ValueError(
                    "entry keyed %r carries target %r" % (name, entry.target)
                )
            if name not in TARGETS:
                raise UnknownTargetError(
                    "%r is keyed by a name that is not canonical; use "
                    "resolve_target() first" % (name,)
                )
        self.unmodelled = dict(unmodelled or {})

    @property
    def targets(self):
        return tuple(sorted(self.entries))

    def ki(self, target):
        return self.entries[resolve_target(target)].ki_nM

    def efficacy(self, target):
        return self.entries[resolve_target(target)].efficacy

    def __contains__(self, target):
        return is_modelled(target) and resolve_target(target) in self.entries

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        return "BindingFingerprint(%r, %d targets: %s)" % (
            self.compound, len(self.entries), ", ".join(self.targets)
        )

    @classmethod
    def from_pdsp_rows(cls, rows, efficacy, compound=None, pools=None,
                       combine="geometric-mean"):
        """Build a fingerprint from PDSP-shaped rows plus an efficacy table.

        Parameters
        ----------
        rows : sequence of `PDSPRow` or of mappings
            A mapping is read by the `PDSPRow` field names.
        efficacy : mapping
            `{target: efficacy}` or `{target: (efficacy, source)}`. Keys may
            be any accepted alias. Every modelled target the rows cover must
            appear, or `MissingEfficacyError` names the ones that do not.
            An efficacy given for a target the rows do not cover is an error
            too, because it claims a functional measurement for a target with
            no affinity behind it.
        compound : str, optional
            Defaults to the rows' `test_ligand`, which must then be
            unambiguous: a fingerprint is one compound's.
        pools : mapping, optional
            `{target: "orthosteric" | "channel"}` for the targets whose pool
            the caller means to declare. Defaults to orthosteric everywhere;
            see the module docstring.
        combine : "geometric-mean" or "min" or "max"
            How several rows for the SAME target are reduced to one Ki.
            Geometric mean by default, which is the right average for a
            quantity whose spread is log-normal and is what this codebase
            already does for a `(low, high)` pair
            (`cannabinoid.cb1_kd`, `acetylcholine.muscarinic_kd`). `min` and
            `max` exist so a caller can report the range's effect rather than
            assert the mean, as `cannabinoid.AFFINITY_CHOICES` does.
        """
        rows = [r if isinstance(r, PDSPRow) else PDSPRow(**dict(r)) for r in rows]
        if not rows:
            raise ValueError("a fingerprint needs at least one row")

        names = {row.test_ligand for row in rows}
        if compound is None:
            if len(names) != 1:
                raise ValueError(
                    "these rows cover %d compounds (%s); a fingerprint is one "
                    "compound's, so pass compound= or split the table"
                    % (len(names), ", ".join(sorted(names)))
                )
            compound = names.pop()

        if combine not in ("geometric-mean", "min", "max"):
            raise ValueError(
                "combine must be 'geometric-mean', 'min' or 'max'; got %r"
                % (combine,)
            )

        by_target, unmodelled = {}, {}
        for row in rows:
            if not is_modelled(row.receptor):
                unmodelled.setdefault(
                    row.receptor,
                    "no receptor or transporter module in this codebase "
                    "carries this target",
                )
                continue
            by_target.setdefault(resolve_target(row.receptor), []).append(row)

        efficacy_by_target = _fold_aliases(efficacy, "efficacy", _as_efficacy)

        if not by_target:
            raise UnknownTargetError(
                "none of %s's %d row(s) name a target this model carries "
                "(%s), so there is no fingerprint to build. An empty "
                "fingerprint would predict nothing while looking like a "
                "prediction."
                % (compound, len(rows), ", ".join(sorted(unmodelled)))
            )

        missing = sorted(set(by_target) - set(efficacy_by_target))
        if missing:
            raise MissingEfficacyError(
                "no efficacy given for %s. A Ki says how tightly a compound "
                "binds and nothing about what the bound receptor does, and "
                "no binding database publishes efficacy, so it has to come "
                "from a functional assay and be passed in. See "
                "fingerprint.py's module docstring."
                % ", ".join(missing)
            )
        spurious = sorted(set(efficacy_by_target) - set(by_target))
        if spurious:
            raise ValueError(
                "efficacy given for %s, which no row provides an affinity "
                "for. An efficacy with no Ki behind it has no place to act."
                % ", ".join(spurious)
            )

        pools = _fold_aliases(pools, "pools", lambda value: value)
        entries = {}
        for target, target_rows in by_target.items():
            kis = np.array([row.ki_nM for row in target_rows], dtype=float)
            if np.any(kis <= 0.0) or not np.all(np.isfinite(kis)):
                raise ValueError(
                    "%s at %s: Ki values must be positive and finite, got %s"
                    % (compound, target, kis.tolist())
                )
            if kis.size == 1:
                # Not `exp(mean(log(x)))` of a single value: that is the same
                # number in exact arithmetic and differs in the last bit in
                # floating point, which would stop a one-row fingerprint from
                # reproducing a hand-built ligand exactly.
                ki_nM = float(kis[0])
            elif combine == "geometric-mean":
                ki_nM = float(np.exp(np.mean(np.log(kis))))
            elif combine == "min":
                ki_nM = float(kis.min())
            else:
                ki_nM = float(kis.max())
            value, efficacy_source = efficacy_by_target[target]
            provenance = {
                field: _unanimous(getattr(row, field) for row in target_rows)
                for field in ("species", "source", "hot_ligand", "reference")
            }
            note = ""
            if len(target_rows) > 1:
                note = "%d determinations combined by %s (%s)" % (
                    len(target_rows), combine,
                    ", ".join("%g" % k for k in kis),
                )
                # A merged Ki whose rows disagree about species, preparation,
                # radioligand or paper cannot claim any one of them, so the
                # field is None and the note carries all of the values. The
                # module docstring says a fingerprint's species is recorded
                # and reported; recording the first of three would not be
                # recording it.
                for field, merged in sorted(provenance.items()):
                    if merged is None:
                        values = sorted(
                            {str(getattr(row, field)) for row in target_rows}
                        )
                        note += "; %s differs across them (%s)" % (
                            field, ", ".join(values)
                        )
            entries[target] = FingerprintEntry(
                target=target,
                ki_nM=ki_nM,
                efficacy=value,
                efficacy_source=efficacy_source,
                pool=pools.get(target, "orthosteric"),
                note=note,
                **provenance
            )
        return cls(compound=compound, entries=entries, unmodelled=unmodelled)


class LigandSet(object):
    """One compound's ligands across every modelled target it binds.

    `ligands` maps a canonical target name to a `receptor.Ligand` (for a
    receptor) or a `transport.TransporterLigand` (for a transporter).
    `provenance` maps the same keys to the `(kon, kon_provenance)` pair the
    ligand was built with, `(None, "not-needed")` at a transporter.
    """

    def __init__(self, compound, ligands, provenance, unmodelled=None):
        self.compound = str(compound)
        self.ligands = dict(ligands)
        self.provenance = dict(provenance)
        self.unmodelled = dict(unmodelled or {})

    @property
    def targets(self):
        return tuple(sorted(self.ligands))

    def __getitem__(self, target):
        return self.ligands[resolve_target(target)]

    def __len__(self):
        return len(self.ligands)

    def __repr__(self):
        return "LigandSet(%r, %d targets)" % (self.compound, len(self.ligands))


def ligand_set(fingerprint, targets=None):
    """The compound's ligand at every target in `fingerprint`.

    This is P8.1's core operation: one Ki table in, one ligand per target
    out, each carrying the affinity exactly and the timescale of its target's
    `kon`. `targets` restricts the result to a subset, resolved through the
    alias table, and raises `UnknownTargetError` for a name this model does
    not carry -- unlike building a fingerprint, where an unrecognised target
    is recorded rather than refused.
    """
    if targets is None:
        wanted = fingerprint.targets
    else:
        wanted = tuple(resolve_target(t) for t in targets)
        absent = [t for t in wanted if t not in fingerprint.entries]
        if absent:
            raise KeyError(
                "%s has no affinity for %s"
                % (fingerprint.compound, ", ".join(sorted(absent)))
            )

    ligands, provenance = {}, {}
    for target in wanted:
        entry = fingerprint.entries[target]
        spec = TARGETS[target]
        if spec.is_transporter:
            ligands[target] = TransporterLigand(
                name=fingerprint.compound,
                ki=entry.ki_nM,
                release_ec50=entry.release_ec50,
                release_emax=entry.release_emax,
            )
            provenance[target] = (None, spec.kon_provenance)
            continue
        kon, kon_provenance = spec.kon, spec.kon_provenance
        if entry.pool == "channel":
            kon, kon_provenance = _channel_kon(spec)
        ligands[target] = ligand_from_ki(
            name=fingerprint.compound,
            ki_nM=entry.ki_nM,
            kon=kon,
            efficacy=entry.efficacy,
        )
        provenance[target] = (kon, kon_provenance)
    return LigandSet(
        compound=fingerprint.compound,
        ligands=ligands,
        provenance=provenance,
        unmodelled=fingerprint.unmodelled,
    )


#: The four targets whose site carries a channel-blocker pool, and the
#: `(kon, provenance)` a ligand declared into it takes. FOUR, not one: the two
#: nicotinic sites carry a mecamylamine pool, and `DesensitizingReceptorSite`
#: deliberately does not subclass `ChannelBlockingReceptorSite` (see
#: `receptor.py`), so an isinstance check misses them.
#:
#: NMDA is the one place a fingerprint ligand gets a MEASURED rate:
#: [Dravid2007]'s MK-801 association rate at the channel site. AMPA's site
#: class carries the pool and the module puts nothing in it, and the two
#: nicotinic pools have no published kinetics, so all three keep their site's
#: illustrative value and say so.
_CHANNEL_KON = {
    "NMDA": (_glu.NMDA_CHANNEL_KON, "measured-at-site"),
    "AMPA": (_glu.ILLUSTRATIVE_KON, "illustrative"),
    "alpha4beta2": (_ach.ILLUSTRATIVE_KON, "illustrative"),
    "alpha7": (_ach.ILLUSTRATIVE_KON, "illustrative"),
}


def _channel_kon(spec):
    """`(kon, provenance)` for a ligand declared into a site's channel pool.

    A target whose site has no such pool raises, because a pool a site does
    not have is a caller error and not something to approximate.
    """
    if spec.name not in _CHANNEL_KON:
        raise ValueError(
            "%s has no channel-blocker pool, so pool='channel' has no meaning "
            "there. The sites that do: %s"
            % (spec.name, ", ".join(sorted(_CHANNEL_KON)))
        )
    return _CHANNEL_KON[spec.name]


# ---------------------------------------------------------------------------
# Wiring a fingerprint ligand into a system
# ---------------------------------------------------------------------------


def _rebuilt_site(site, orthosteric, blockers, k_des=None, k_rec=None):
    """`site`'s class, rebuilt over new ligand lists.

    Dispatches on the site class rather than on a flag, because the four
    classes differ in what a "ligand list" even means: a plain `ReceptorSite`
    has one pool, `ChannelBlockingReceptorSite` two, `DesensitizingReceptor
    Site` two plus a per-orthosteric-ligand rate pair, and
    `GlutamatergicTransmissionSite` adds a synaptic share. Everything that is
    not a ligand is carried across unchanged.
    """
    if isinstance(site, _glu.GlutamatergicTransmissionSite):
        return _glu.GlutamatergicTransmissionSite(
            name=site.name, ligands=orthosteric,
            synaptic_share=site.synaptic_share, channel_blockers=blockers,
        )
    if isinstance(site, DesensitizingReceptorSite):
        return DesensitizingReceptorSite(
            name=site.name, ligands=orthosteric, k_des=k_des, k_rec=k_rec,
            channel_blockers=blockers,
        )
    if isinstance(site, ChannelBlockingReceptorSite):
        return ChannelBlockingReceptorSite(
            name=site.name, ligands=orthosteric, channel_blockers=blockers,
        )
    if blockers:
        raise ValueError(
            "%r is a plain ReceptorSite and has no channel-blocker pool"
            % site.name
        )
    return ReceptorSite(name=site.name, ligands=orthosteric)


def _split_pools(site):
    """`(orthosteric, blockers)` for any of the four site classes."""
    orthosteric = list(getattr(site, "orthosteric_ligands", site.ligands))
    blockers = list(getattr(site, "channel_blockers", []))
    return orthosteric, blockers


def ligand_occupancy(site, states, ligand_index):
    """The fraction of `site`'s pool held by one ligand, over time.

    `states` is that site's whole state block, `(n_states, n_times)`;
    `ligand_index` indexes `site.ligands`. Returns `(n_times,)`.

    This is not `states[ligand_index]`, and the two differ in both of the
    ways a site's state layout can differ from its ligand layout:

    * **A `DesensitizingReceptorSite` holds a ligand in TWO states.** A
      nicotinic agonist that has driven the receptor into the desensitized
      conformation is still bound and, in that class's own words, "occupies
      the pool (so it still competes)". Reporting only the conducting state
      would understate what fraction of the receptor the drug is holding, by
      up to `1 + k_des/k_rec`, and it is the OCCUPANCY that a fingerprint
      prediction is about: what the site does with it is `activation`, which
      correctly counts only the conducting fraction.
    * **A blocker's state index is not its ligand index.** The ligand order
      is orthosteric-then-blocker, and the state order for a desensitizing
      site is bound, then desensitized, then blocked, so blocker `j` is
      ligand `p + j` and state `2p + j`. `split_states` is the site's own
      answer to this and is what this uses.
    """
    states = np.asarray(states, dtype=float)
    if isinstance(site, DesensitizingReceptorSite):
        p = site.n_orthosteric
        bound, desensitized, blocked = states[0:p], states[p:2 * p], states[2 * p:]
        if ligand_index < p:
            return bound[ligand_index] + desensitized[ligand_index]
        return blocked[ligand_index - p]
    # For `ReceptorSite` and for `ChannelBlockingReceptorSite` (whose layout
    # is bound-then-blocked with one state per ligand) the two orders
    # coincide, so the row IS the ligand index.
    return states[ligand_index]


def add_to_system(system, ligand, pk_model, pool="orthosteric",
                  k_des=None, k_rec=None, on_duplicate="error"):
    """A copy of `system` with one more ligand at its receptor site.

    The system's coupling, connection modulation, name, transmitter sources
    and every existing ligand and PK model come through unchanged; see the
    module docstring on why this is the only way a fingerprint may touch a
    system.

    `pk_model` is the new ligand's own PK model. The returned system's
    `pk_models` are reordered to match the site's ligand order, which the
    engine requires and which changes when an orthosteric ligand is inserted
    into a site that also has channel blockers.

    `k_des`/`k_rec` are the desensitization and recovery rates, meaningful
    only at a `DesensitizingReceptorSite`, and `None` means NOT STATED, which
    is what a Ki table supports. Three cases, none of them silent: a stated
    rate always wins; an unstated one on an ADDED ligand is 0, the class's own
    way of saying "occupies the orthosteric site without driving the
    conformational change"; and an unstated one on a REPLACED ligand keeps the
    module's own measured rate rather than zeroing it.

    `on_duplicate`
        `"error"` (default) refuses to add a ligand whose name is already at
        the site. `"replace"` substitutes it, which is what a round-trip test
        needs -- rebuilding a hand-built ligand from its own published Ki and
        checking the scenario still behaves -- and is otherwise a way to
        overwrite a calibrated ligand, so it is never the default.
    """
    if pool not in POOLS:
        raise ValueError("pool must be one of %s; got %r" % (POOLS, pool))
    if on_duplicate not in ("error", "replace"):
        raise ValueError("on_duplicate must be 'error' or 'replace'")

    site = system.receptor_site
    orthosteric, blockers = _split_pools(site)
    pk_by_name = {
        lig.name: model for lig, model in zip(site.ligands, system.pk_models)
    }
    desensitizing = isinstance(site, DesensitizingReceptorSite)
    rates = (
        {
            lig.name: (float(site.k_des[i]), float(site.k_rec[i]))
            for i, lig in enumerate(orthosteric)
        }
        if desensitizing
        else {}
    )

    already = None
    if any(lig.name == ligand.name for lig in orthosteric):
        already = "orthosteric"
    elif any(lig.name == ligand.name for lig in blockers):
        already = "channel"

    if already is not None:
        if on_duplicate == "error":
            raise ValueError(
                "%r is already a ligand at %r. Pass on_duplicate='replace' to "
                "substitute it, which overwrites a calibrated ligand and is "
                "why it is not the default." % (ligand.name, site.name)
            )
        if already != pool:
            raise ValueError(
                "%r is already in %r's %s pool, so it cannot be replaced by a "
                "%s one. Moving a ligand between pools is a claim about its "
                "mechanism, not a substitution."
                % (ligand.name, site.name, already, pool)
            )
        # Substituted IN PLACE rather than removed and re-appended, so the
        # site's ligand order -- and therefore the engine's state layout and
        # `doses` order -- is exactly what it was. That is what lets a round
        # trip be compared row by row against the scenario it rebuilds.
        destination = orthosteric if pool == "orthosteric" else blockers
        destination[[lig.name for lig in destination].index(ligand.name)] = ligand
    elif pool == "channel":
        if not hasattr(site, "channel_blockers"):
            # Not an isinstance check: `DesensitizingReceptorSite` carries a
            # channel-blocker pool and deliberately does NOT subclass
            # `ChannelBlockingReceptorSite` (receptor.py explains why their
            # state layouts interleave differently), so testing for the class
            # would refuse the two nicotinic sites that do have one.
            raise ValueError(
                "%r has no channel-blocker pool, so pool='channel' has no "
                "meaning there" % site.name
            )
        blockers.append(ligand)
    else:
        orthosteric.append(ligand)

    if desensitizing and pool == "orthosteric":
        stated = None if k_des is None and k_rec is None else (
            float(k_des or 0.0), float(k_rec or 0.0)
        )
        if stated is not None:
            # A stated rate always wins, including over an existing ligand's,
            # so a replacement cannot silently keep the old one.
            rates[ligand.name] = stated
        elif ligand.name not in rates:
            # Added, and nothing stated: k_des = 0 is the class's own way of
            # saying "occupies the orthosteric site without driving the
            # conformational change", which is all a Ki table supports.
            rates[ligand.name] = (0.0, 0.0)
    pk_by_name[ligand.name] = pk_model

    extra = {}
    if desensitizing:
        extra["k_des"] = [rates[lig.name][0] for lig in orthosteric]
        extra["k_rec"] = [rates[lig.name][1] for lig in orthosteric]
    new_site = _rebuilt_site(site, orthosteric, blockers, **extra)

    return ReceptorSystem(
        pk_models=[pk_by_name[lig.name] for lig in new_site.ligands],
        receptor_site=new_site,
        coupling=system.coupling,
        connection_modulation=system.connection_modulation,
        name=system.name,
        transmitter_sources=getattr(system, "transmitter_sources", None),
    )


def add_to_transporter_system(system, ligand, pk_model, on_duplicate="error"):
    """A copy of a `TransporterSystem` with one more drug at its transporter.

    The transporter's kinetics (`km`/`vmax` or `clearance_rate`), its basal
    concentration, its release rate and its release capacity are carried
    across unchanged; only the drug list grows.
    """
    if on_duplicate not in ("error", "replace"):
        raise ValueError("on_duplicate must be 'error' or 'replace'")
    transporter = system.transporter
    ligands = list(transporter.ligands)
    pk_by_name = {
        lig.name: model for lig, model in zip(ligands, system.pk_models)
    }
    if ligand.name in pk_by_name:
        if on_duplicate == "error":
            raise ValueError(
                "%r is already a drug at %r; pass on_duplicate='replace' to "
                "substitute it" % (ligand.name, transporter.name)
            )
        # In place, for the same reason as `add_to_system`.
        ligands[[l.name for l in ligands].index(ligand.name)] = ligand
    else:
        ligands.append(ligand)
    pk_by_name[ligand.name] = pk_model

    rebuilt = MonoamineTransporter(
        name=transporter.name,
        transmitter=transporter.transmitter,
        ligands=ligands,
        km=transporter.km,
        vmax=transporter.vmax,
        clearance_rate=transporter.clearance_rate,
        basal=transporter.basal,
        release_rate=transporter.release_rate,
        release_capacity=transporter.release_capacity,
    )
    return TransporterSystem(
        pk_models=[pk_by_name[lig.name] for lig in rebuilt.ligands],
        transporter=rebuilt,
        name=system.name,
    )


def receptor_systems_for(fingerprint, pk_model, targets=None,
                         on_duplicate="error"):
    """`{target: system}`: the compound wired into every target it binds.

    Each value is an `engine.ReceptorSystem` or `engine.TransporterSystem`
    ready to hand to `engine.Simulation`, carrying its module's own coupling.
    This is the object P8.2 needs in order to compose several targets onto
    one whole-brain network; `predict` below uses the same systems without a
    circuit.

    `pk_model` is the compound's PK model, one object shared by every target,
    because a compound has one disposition and several receptors.
    """
    ligands = ligand_set(fingerprint, targets=targets)
    systems = {}
    for target, ligand in ligands.ligands.items():
        spec = TARGETS[target]
        base = spec.system()
        if spec.is_transporter:
            systems[target] = add_to_transporter_system(
                base, ligand, pk_model, on_duplicate=on_duplicate
            )
            continue
        entry = fingerprint.entries[target]
        systems[target] = add_to_system(
            base, ligand, pk_model, pool=entry.pool,
            k_des=entry.k_des, k_rec=entry.k_rec, on_duplicate=on_duplicate,
        )
    return systems


# ---------------------------------------------------------------------------
# The receptor layer, integrated without a circuit
# ---------------------------------------------------------------------------


class _CircuitFreeBlock(object):
    """One system's PK-plus-binding right-hand side, with no circuit attached.

    This is `engine.Simulation` with the network, the drive and the connection
    modulation removed, and nothing else changed: the same per-ligand PK
    models through `pk.as_ode_pk_model`, the same
    `receptor_concentration_factor` correction that turns a plasma
    concentration into the unbound brain concentration a receptor binds
    (P1.5), and the site's or transporter's own right-hand side. It exists
    because a receptor-layer prediction has nothing to say about a circuit and
    `Simulation` requires one.

    It is a SECOND implementation of a right-hand side this codebase already
    has, so it is checked against the first rather than trusted:
    `test_fingerprint.py::test_the_circuit_free_rhs_is_the_engines_rhs`
    evaluates both at the same state and requires them to agree exactly, which
    is a stronger statement than two integrations landing close.

    `state_rhs(block_state, concentrations) -> d block_state/dt` is
    `ReceptorSite.occupancy_rhs` or `MonoamineTransporter.transmitter_rhs`.
    """

    def __init__(self, pk_models, n_states, state_rhs):
        self.models = [as_ode_pk_model(model) for model in pk_models]
        self.n_states = int(n_states)
        self.state_rhs = state_rhs
        self.blocks, rows, factors = [], [], []
        cursor = 0
        for model in self.models:
            width = int(model.n_states)
            self.blocks.append((cursor, cursor + width))
            rows.append(cursor)
            factors.append(
                float(getattr(model, "receptor_concentration_factor", 1.0))
            )
            cursor += width
        self.d = cursor
        self.rows = np.array(rows, dtype=int)
        self.factors = np.array(factors, dtype=float)

    def rhs(self, t, y):
        derivative = np.empty(self.d + self.n_states)
        for model, (start, stop) in zip(self.models, self.blocks):
            derivative[start:stop] = model.concentration_rhs(y[start:stop])
        derivative[self.d:] = self.state_rhs(
            y[self.d:], y[self.rows] * self.factors
        )
        return derivative

    def initial_state(self, doses, block_initial):
        pk = (
            np.concatenate(
                [m.initial_state(dose) for m, dose in zip(self.models, doses)]
            )
            if self.models
            else np.zeros(0)
        )
        return np.concatenate([pk, np.asarray(block_initial, dtype=float)])


def _integrate_block(pk_models, doses, n_states, block_rhs, block_initial,
                     t_span, t_eval, method):
    """Integrate one `_CircuitFreeBlock` and unpack it like `Simulation` does."""
    block = _CircuitFreeBlock(pk_models, n_states, block_rhs)
    solution = solve_ivp(
        block.rhs,
        t_span,
        block.initial_state(doses, block_initial),
        method=method,
        t_eval=t_eval,
        max_step=0.1,
    )
    if not solution.success:
        raise RuntimeError("integration failed: %s" % solution.message)
    plasma = solution.y[block.rows]
    return {
        "t": solution.t,
        "plasma_concentrations": plasma,
        "receptor_concentrations": plasma * block.factors[:, None],
        "block": solution.y[block.d:],
    }


class TargetPrediction(object):
    """What one target does under one dose of one compound.

    Attributes
    ----------
    target, ligand, system
    t : (n_times,) array, minutes
    receptor_concentration : (n_times,) array, nM
        What the site actually bound: unbound brain concentration when the PK
        model carries a partition, plasma otherwise. `compartment` says which.
    occupancy : (n_times,) array or None
        The COMPOUND's own fractional occupancy of the site. `None` at a
        transporter, which has no occupancy state -- see `transmitter`.
    activation : (n_times,) array or None
        The whole site's activation, `sum(occupancy_i * efficacy_i)`, in the
        sign convention of that site's own class.
    transmitter : (n_times,) array or None
        Transporter targets only: the extracellular transmitter concentration
        in nM, which is what a transporter's whole output is.
    """

    def __init__(self, target, ligand, system, t, receptor_concentration,
                 compartment, occupancy=None, activation=None,
                 transmitter=None):
        self.target = target
        self.ligand = ligand
        self.system = system
        self.t = t
        self.receptor_concentration = receptor_concentration
        self.compartment = compartment
        self.occupancy = occupancy
        self.activation = activation
        self.transmitter = transmitter

    @property
    def peak_occupancy(self):
        return None if self.occupancy is None else float(self.occupancy.max())

    @property
    def time_of_peak_occupancy_min(self):
        if self.occupancy is None:
            return None
        return float(self.t[int(np.argmax(self.occupancy))])

    @property
    def peak_receptor_concentration_nM(self):
        return float(self.receptor_concentration.max())

    def __repr__(self):
        if self.occupancy is None:
            return "TargetPrediction(%r, transporter)" % (self.target,)
        return "TargetPrediction(%r, peak occupancy %.3f at %.1f min)" % (
            self.target, self.peak_occupancy, self.time_of_peak_occupancy_min
        )


class CompoundPrediction(object):
    """`predict`'s result: the receptor layer, per target, plus its caveats."""

    def __init__(self, compound, dose, route, species, body_weight_kg,
                 pk_model, brain_partition_status, targets, unmodelled):
        self.compound = compound
        self.dose = dose
        self.route = route
        self.species = species
        self.body_weight_kg = body_weight_kg
        self.pk_model = pk_model
        self.brain_partition_status = brain_partition_status
        self.targets = dict(targets)
        self.unmodelled = dict(unmodelled)

    def __getitem__(self, target):
        return self.targets[resolve_target(target)]

    def occupancy_ranking(self):
        """`[(target, peak occupancy), ...]`, highest first.

        The receptor-layer shape of a compound at one dose: which of its
        targets it actually reaches. Transporter targets are excluded because
        they have no occupancy to rank.
        """
        ranked = [
            (name, prediction.peak_occupancy)
            for name, prediction in self.targets.items()
            if prediction.occupancy is not None
        ]
        return sorted(ranked, key=lambda pair: -pair[1])

    def __repr__(self):
        return "CompoundPrediction(%r, %g %s %s, %d targets)" % (
            self.compound, self.dose, self.route, self.species,
            len(self.targets)
        )


def predict(fingerprint, dose, route, species, body_weight_kg, targets=None,
            pk_model=None, t_span=(0.0, 120.0), n_times=241,
            on_duplicate="error", method="LSODA"):
    """Receptor-layer prediction for a compound from its binding fingerprint.

    The chain, all of it already in this codebase and none of it re-derived
    here::

        dose (mg/kg), route, species
            --drug_pk.pk_model_for, i.e. the Phase 1 bridge-->
        plasma concentration C(t)
            --brain_partition, i.e. P1.5-->
        unbound brain concentration
            --ReceptorSite.occupancy_rhs at koff = kon * Ki-->
        the compound's occupancy, and the site's activation

    and it stops there on purpose; see the module docstring's "Where this
    stops".

    Parameters
    ----------
    dose : float
        In whatever unit the PK model doses in, exactly as
        `engine.Simulation.run`: mg/kg for a dose-administered model, an
        achieved bolus concentration in nM for a bare `pk.OneCompartmentPK`.
        0 means "no drug" for either.
    route : one of `pk.ROUTES`
    species : str
        Selects the PK parameter set and the brain partition, NOT the binding
        constants. See the module docstring.
    body_weight_kg : float
        Never defaulted, following `drug_pk`'s rule 3.
    pk_model : optional
        Supply the compound's PK model directly instead of looking it up.
        The only way to run a compound `drug_pk` has no parameter set for,
        and the caller then owns that model's provenance.
    on_duplicate : "error" or "replace"
        Passed to `add_to_system`; `"replace"` is what reproduces a
        hand-built ligand from its own published Ki.
    """
    supplied_pk_model = pk_model
    if pk_model is None:
        pk_model = _drug_pk.pk_model_for(
            fingerprint.compound, species, route, body_weight_kg=body_weight_kg
        )
    if supplied_pk_model is None:
        partition_status = _drug_pk.brain_partition_status(
            fingerprint.compound, species
        )
    else:
        # A supplied model's partition is the caller's, so `drug_pk`'s record
        # for this compound describes something else entirely -- and for the
        # arbitrary compound this API exists to accept, that record is always
        # "absent" while the supplied model may perfectly well carry a
        # partition. Reporting the lookup here would state the opposite of
        # what the run did. The model itself is the only witness, so it is
        # what is read.
        compartment = getattr(
            supplied_pk_model, "receptor_compartment", "unknown"
        )
        partition_status = (
            "receptor-drivable" if compartment == "unbound brain"
            else "caller-supplied",
            "the PK model was supplied by the caller, who owns its "
            "provenance; it drives the receptor with %s" % compartment,
        )

    systems = receptor_systems_for(
        fingerprint, pk_model, targets=targets, on_duplicate=on_duplicate
    )
    t_eval = np.linspace(float(t_span[0]), float(t_span[1]), int(n_times))

    predictions = {}
    for target, system in systems.items():
        spec = TARGETS[target]
        if spec.is_transporter:
            transporter = system.transporter
            names = [lig.name for lig in transporter.ligands]
            doses = [
                dose if name == fingerprint.compound else 0.0 for name in names
            ]
            result = _integrate_block(
                system.pk_models, doses, transporter.n_states,
                transporter.transmitter_rhs, transporter.initial_state(),
                t_span, t_eval, method,
            )
            index = names.index(fingerprint.compound)
            predictions[target] = TargetPrediction(
                target=target,
                ligand=transporter.ligands[index],
                system=system,
                t=result["t"],
                receptor_concentration=result["receptor_concentrations"][index],
                compartment=getattr(
                    system.pk_models[index], "receptor_compartment", "unknown"
                ),
                transmitter=result["block"][0],
            )
            continue

        if getattr(system, "transmitter_sources", None):
            # `_CircuitFreeBlock` has no transporter or projection pools, so a
            # ligand whose concentration is supposed to come from one would
            # silently bind its own PK model's zero instead. No system in
            # `TARGETS` declares one today (a test asserts it); this refuses
            # rather than degrades if one ever does.
            raise NotImplementedError(
                "%r sources a ligand from a transmitter pool, which needs a "
                "circuit; predict() runs the receptor layer alone. Build the "
                "system with receptor_systems_for() and run it in an "
                "engine.Simulation instead." % system.name
            )
        site = system.receptor_site
        names = [lig.name for lig in site.ligands]
        doses = [dose if name == fingerprint.compound else 0.0 for name in names]
        result = _integrate_block(
            system.pk_models, doses, site.n_states,
            site.occupancy_rhs, np.zeros(site.n_states),
            t_span, t_eval, method,
        )
        index = names.index(fingerprint.compound)
        block = result["block"]
        activation = np.array(
            [site.activation(block[:, k]) for k in range(block.shape[1])]
        )
        predictions[target] = TargetPrediction(
            target=target,
            ligand=site.ligands[index],
            system=system,
            t=result["t"],
            receptor_concentration=result["receptor_concentrations"][index],
            compartment=getattr(
                system.pk_models[index], "receptor_compartment", "unknown"
            ),
            occupancy=ligand_occupancy(site, block, index),
            activation=activation,
        )

    return CompoundPrediction(
        compound=fingerprint.compound,
        dose=float(dose),
        route=route,
        species=species,
        body_weight_kg=body_weight_kg,
        pk_model=pk_model,
        brain_partition_status=partition_status,
        targets=predictions,
        unmodelled=fingerprint.unmodelled,
    )


def target_report():
    """The `TARGETS` table as text: every target's `kon` and its standing.

    The table a reader needs in order to know how much a fingerprint-derived
    rate constant is worth at each target, printed rather than asserted.
    """
    lines = [
        "Binding-fingerprint targets (P8.1)",
        "=" * 70,
        "",
        "koff = kon * Ki at every receptor target; the Ki fixes the ratio and",
        "neither rate, so the kon column is where the assumption lives.",
        "",
    ]
    for provenance in KON_PROVENANCES:
        members = sorted(
            name for name, spec in TARGETS.items()
            if spec.kon_provenance == provenance
        )
        if not members:
            continue
        lines.append("%s (%d)" % (provenance, len(members)))
        lines.append("-" * 70)
        for name in members:
            spec = TARGETS[name]
            kon = "n/a" if spec.kon is None else "%.6g nM^-1 min^-1" % spec.kon
            lines.append("  %-12s kon = %s" % (name, kon))
            lines.append("      density gene: %s%s"
                         % (spec.gene, " (proxy)" if spec.gene_note else ""))
            lines.append("      %s" % spec.kon_source)
            if spec.gene_note:
                lines.append("      gene: %s" % spec.gene_note)
            if spec.note:
                lines.append("      note: %s" % spec.note)
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "UnknownTargetError",
    "MissingEfficacyError",
    "KON_PROVENANCES",
    "POOLS",
    "TARGETS",
    "TargetSpec",
    "PDSPRow",
    "FingerprintEntry",
    "BindingFingerprint",
    "LigandSet",
    "TargetPrediction",
    "CompoundPrediction",
    "normalize_target",
    "resolve_target",
    "is_modelled",
    "ligand_set",
    "add_to_system",
    "add_to_transporter_system",
    "receptor_systems_for",
    "predict",
    "target_report",
]
