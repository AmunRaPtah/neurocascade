"""One already-sourced receptor system, run on the human whole-brain network.

This is the last join in the human data layer. `human/maps.py` turns one
gene's AHBA row into the `{node: gain * rho[region]}` coupling dict;
`human/build.py` turns the ENIGMA matrix into a species-neutral
`CircuitNetwork`. This module takes those two, plus an EXISTING receptor
system (a `ReceptorSite` and its per-ligand PK models, from `scenarios.py`
or a P5 module), and composes them into the `engine.Simulation` the engine
already integrates.

It adds no physics and no numbers. Every layer's provenance is recorded in
`HumanScenarioReport`, and the one number this layer has never earned - the
`mechanism_gain` - is taken from the caller and reported as unfitted for
humans rather than hidden. That split is the same one `human/maps.py`
states, restated here because a full Simulation is the first object where a
reader might mistake the result for a calibrated human prediction:

    sourced, human       the connectome (ENIGMA HCP), the expression map
                         (AHBA), and whatever the caller's receptor system
                         itself sourced - each species-labelled by its own
                         module, not relabelled here
    ILLUSTRATIVE         the node parameters (mouse `DEFAULT_NODE`), the
                         operating gain `rho`, and the caller-supplied
                         `mechanism_gain`, which no human data has ever
                         fitted in this repository

A result produced here is therefore a human-DATA-DRIVEN whole-brain response
pattern: the regions that move most are fixed by the human connectome and
the human expression map, while the magnitude of the movement is fixed by
an unfitted gain and must be read as illustrative. The two facts travel
together in the report rather than being allowed to blur.

    from neurocascade import human, scenarios

    expr = human.load_expression()
    scenario = human.whole_brain_simulation(
        expr, "OPRM1",
        pk_models=scenarios.mu_opioid_pk_models_calibrated(),
        receptor_site=scenarios.mu_opioid_receptor_site_calibrated(),
        mechanism_gain=scenarios.MU_OPIOID_MECHANISM_GAIN,
    )
    print(scenario.report.summary())
    result = scenario.simulation.run(
        dose_concentrations=[148.6, 0.0], t_span=(0, 60), t_eval=...
    )
"""

from .. import stability
from .. import species_gate
from ..engine import Simulation
from . import build as _build
from . import maps as _maps

__all__ = [
    "HumanScenarioReport",
    "HumanScenario",
    "whole_brain_simulation",
]

#: What the report says about the one number this layer does not supply. It
#: is a fixed statement rather than a parameter because the standing of a
#: human mechanism gain is itself a fixed fact of this repository: none has
#: ever been fitted.
MECHANISM_GAIN_NOTE = (
    "caller-supplied; no human mechanism_gain has ever been fitted in this "
    "repository, so its magnitude is ILLUSTRATIVE and its provenance is the "
    "caller's"
)


class HumanScenarioReport(object):
    """Where every number in one composed human scenario came from.

    A record, not a calculation: each field is filled once at build time so
    that the provenance of the Simulation cannot drift from what actually
    ran, the same reason `human/build.BuildReport` exists.
    """

    def __init__(
        self,
        symbol,
        site_name,
        ligand_names,
        n_pk_models,
        mechanism_gain,
        mechanism_gain_note,
        connectome_key,
        n_nodes,
        n_cortical,
        n_subcortical,
        weight_scale,
        operating_gain,
        normalization,
        missing,
        excluded_regions,
    ):
        self.symbol = symbol
        self.site_name = site_name
        self.ligand_names = tuple(ligand_names)
        self.n_pk_models = int(n_pk_models)
        self.mechanism_gain = mechanism_gain
        self.mechanism_gain_note = mechanism_gain_note
        self.connectome_key = connectome_key
        self.n_nodes = int(n_nodes)
        self.n_cortical = int(n_cortical)
        self.n_subcortical = int(n_subcortical)
        self.weight_scale = float(weight_scale)
        self.operating_gain = float(operating_gain)
        self.normalization = normalization
        self.missing = missing
        self.excluded_regions = tuple(excluded_regions)

    @property
    def mechanism_gain_is_scalar(self):
        return isinstance(self.mechanism_gain, (int, float))

    def summary(self):
        gain = self.mechanism_gain
        if isinstance(gain, dict):
            gain_text = "per-node dict over %d nodes" % len(gain)
        else:
            gain_text = "%g" % gain
        lines = [
            "human whole-brain scenario: %s on %s" % (self.site_name, self.symbol),
            "  network: %s, %d nodes (%d cortical + %d subcortical), "
            "weight_scale=%.6g, operating_gain=%.4g (ILLUSTRATIVE)"
            % (
                self.connectome_key,
                self.n_nodes,
                self.n_cortical,
                self.n_subcortical,
                self.weight_scale,
                self.operating_gain,
            ),
            "  expression map: %s, normalization=%r, missing=%r"
            % (self.symbol, self.normalization, self.missing),
            "  receptor system: %s [ligands: %s], %d PK model(s)"
            % (
                self.site_name,
                ", ".join(self.ligand_names),
                self.n_pk_models,
            ),
            "  mechanism_gain: %s (%s)" % (gain_text, self.mechanism_gain_note),
        ]
        if self.excluded_regions:
            lines.append(
                "  regions with no expression, excluded from the coupling: %s"
                % ", ".join(self.excluded_regions)
            )
        return "\n".join(lines)


class HumanScenario(object):
    """A `Simulation`, its network, its coupling, and the provenance of both."""

    def __init__(self, simulation, network, coupling, report):
        self.simulation = simulation
        self.network = network
        self.coupling = dict(coupling)
        self.report = report

    def __repr__(self):
        return "HumanScenario(%r on %r, %d nodes)" % (
            self.report.site_name,
            self.report.symbol,
            self.report.n_nodes,
        )


def whole_brain_simulation(
    expression,
    symbol,
    pk_models,
    receptor_site,
    mechanism_gain,
    connectome=None,
    operating_gain=stability.DEFAULT_OPERATING_GAIN,
    normalization="max",
    missing="exclude",
    labels=None,
):
    """Compose one receptor system onto the stabilized human whole-brain network.

    `pk_models` and `receptor_site` are an existing receptor system (for
    example `scenarios.mu_opioid_pk_models_calibrated()` and
    `scenarios.mu_opioid_receptor_site_calibrated()`); this module does not
    build, source or relabel any of their numbers. `mechanism_gain` is the
    factor `human/maps.py` refuses to supply, and is required here for the
    same reason it is nowhere defaulted there.

    The human network is `stability.stabilize()`d exactly as
    `scripts/check_human_whole_brain_stability.py` does, so the composed
    Simulation rests at a unique, certified, non-saturating baseline. The
    two AHBA-unmatched regions are dropped from the coupling by default
    (`missing="exclude"`) and recorded in the report, following the mouse
    whole-brain composer's precedent of excluding rather than refusing.

    This builder is one of the two places the P11.2 species gate is enforced
    (`fingerprint.predict` is the other). Before anything is wired, every PK
    model passes `species_gate.check_pk_model` and the receptor site passes
    `species_gate.check_receptor_site`, which looks the site up by name in
    the binding-provenance registry: a site whose constants were measured in
    a non-human species raises `UnsourcedParameterError` naming the species,
    and a site with no registered provenance at all refuses rather than
    passing unchecked. Today that means the mu-opioid calibrated site runs
    (human Ki; its dissociation rates' preparation is recorded as
    species-unstated, not asserted human) while the benzodiazepine site --
    rat cortex Kd -- refuses, which is the tree's real state made loud.
    """
    from .matrix import load_connectome

    conn = load_connectome(allow_download=False) if connectome is None else connectome
    built = _build.build_circuit_network(conn)
    stabilized = stability.stabilize(built.network, operating_gain=operating_gain)
    network = stabilized.network

    coupling_result = _maps.whole_brain_coupling(
        expression,
        symbol,
        mechanism_gains=mechanism_gain,
        labels=labels,
        normalization=normalization,
        missing=missing,
    )

    if len(pk_models) != receptor_site.n:
        raise ValueError(
            "need one PK model per ligand on the receptor site: got %d PK "
            "models for %d ligands" % (len(pk_models), receptor_site.n)
        )

    # The P11.2 species gate, before anything is wired: no rodent parameter
    # crosses into a human whole-brain build.
    for pk_model in pk_models:
        species_gate.check_pk_model(pk_model)
    species_gate.check_receptor_site(receptor_site)

    simulation = Simulation(
        pk_models=list(pk_models),
        receptor_site=receptor_site,
        network=network,
        coupling=coupling_result.coupling,
    )

    report = HumanScenarioReport(
        symbol=symbol,
        site_name=receptor_site.name,
        ligand_names=[ligand.name for ligand in receptor_site.ligands],
        n_pk_models=len(pk_models),
        mechanism_gain=mechanism_gain,
        mechanism_gain_note=MECHANISM_GAIN_NOTE,
        connectome_key=conn.dataset_key,
        n_nodes=network.n,
        n_cortical=len(conn.atlas.cortical()),
        n_subcortical=len(conn.atlas.subcortical()),
        weight_scale=stabilized.weight_scale,
        operating_gain=stabilized.operating_gain,
        normalization=normalization,
        missing=missing,
        excluded_regions=coupling_result.excluded,
    )
    return HumanScenario(
        simulation=simulation,
        network=network,
        coupling=coupling_result.coupling,
        report=report,
    )
