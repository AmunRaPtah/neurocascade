"""The human data layer.

This package is the first deliberate move toward a human-first build. The
existing `neurocascade.connectome` package ingests the Allen MOUSE
connectome and CCF ontology; everything in `human/` ingests HUMAN data and
turns it into the same species-neutral `CircuitNetwork` the engine already
integrates. The physics in `circuit.py`, `network.py`, `receptor.py`,
`pk.py` and `engine.py` is unchanged and species-neutral: only the data
layer differs.

Read `sources.py` first: it is the manifest (URLs pinned to a commit,
checksums, citations, and what the matrix's numbers actually mean), and it
is the only place a number about the external data is written down.
`neurotransmitter.py` is the cited per-region sign table that turns the
unsigned dMRI matrix into a signed `CircuitNetwork` (glutamatergic cortex /
hippocampus / thalamus, GABAergic striatum / pallidum, mixed amygdala).

Offline by default
------------------

Same contract as `neurocascade.connectome`: nothing touches the network at
import time, every loader defaults to `allow_download=False`, and a cache
miss raises `MissingCachedFile` telling you to run
`scripts/fetch_human_connectome.py`. Tests run entirely off the small
committed fixtures in `fixtures/`, which are real slices of the verified
files.

    from neurocascade import human

    conn = human.load_connectome()          # verified cache
    built = human.build_circuit_network(conn)
    print(built.report.summary())

What has been built here
------------------------

This track ingests one real, citable human structural connectivity matrix -
the ENIGMA Toolbox's group-average Desikan-Killiany (68 cortical + 14
subcortical) diffusion-MRI tractography matrix - and builds a signed
`CircuitNetwork` from it. The sign is assigned per source region from cited
transmitter identity (`neurotransmitter.py`), the human analogue of
`connectome/neurotransmitter.py`. Three honest caveats are front-and-center
rather than hidden:

* dMRI tractography is UNDIRECTED and UNSIGNED. The mouse matrices are
  directed by tracer injection; the human matrix is symmetric and carries
  no direction, so each symmetric weight is entered twice and signed once
  per direction (`build.py` states this convention out loud).
* The sign is a per-source projection-neuron transmitter identity, not a
  measurement: cortex / hippocampus / thalamus are glutamatergic, striatum
  / pallidum are GABAergic, and the coarse whole-amygdala region is MIXED
  and left unsigned (excluded, or a hard error with
  `unsigned_sources="raise"`).
* The Desikan-Killiany atlas is a cortex-centred parcellation; the 14
  subcortical regions are the coarse classic nuclei, not the fine
  subdivisions the mouse CCF provides.
"""

from . import atlas, build, d2_ec50, d2_effect, d2_in_vitro, dataset_manifest, directed_prior, dmri_fallback, dose_to_occupancy, ei_time_constants, expression, fetch, fixtures, kp_uu, maps, matrix, neurotransmitter, occupancy_validation, pk_parameters, pk_validation, receptor_parameters, provenance, scenario, sources
from .atlas import HumanAtlas, Region, atlas_from_labels, parse_label
from .build import (
    BuildReport,
    HumanCircuit,
    build_circuit_network,
)
from .pk_parameters import HUMAN_PK_RECORDS, HumanPKRecord, human_pk_records
from .kp_uu import HUMAN_KP_UU_RECORDS, KpUURecord, kp_uu_records, transfer_kp_uu
from .pk_validation import HUMAN_PK_VALIDATION, KNOWN_FAILURE_MODES, known_failure_mode, validation_outcomes
from .receptor_parameters import HUMAN_RECEPTOR_PARAMETERS, HumanReceptorParameter, human_receptor_parameters, require_human_parameter
from .mor_effect import HUMAN_MOR_RESERVE_PRIOR, MORReserveModel
from .behavioral_endpoints import HUMAN_ENDPOINTS, HumanEndpoint, endpoint_score, human_endpoints
from .ke0 import HUMAN_KE0_RECORDS, HumanKe0, human_ke0
from .dose_response import HumanDoseResponseFit, HumanDoseResponseObservation, fit_human_log_linear
from .endpoint_status import ENDPOINT_STATUS, EndpointStatus, endpoint_status, require_human_endpoint
from .d2_ec50 import (
    D2_PLASMA_EC50,
    D2PlasmaEC50,
    EC50Comparison,
    EC50Fit,
    EC50_PROVENANCE_CLASSES,
    EMAX_CONVENTIONS,
    HALOPERIDOL_D2_PLASMA_EC50,
    HALOPERIDOL_D2_PLASMA_EC50_nM,
    HALOPERIDOL_PLASMA_FREE_FRACTION,
    PLASMA_BASES,
    compare_with_reported_relations,
    d2_plasma_ec50,
    ec50_landscape,
    fit_plasma_ec50,
    registered_d2_drugs,
)
from .dose_to_occupancy import (
    ChainLayer,
    OccupancyPrediction,
    PROVENANCE_LEVELS,
    WeightScaling,
    default_times_min,
    predict_occupancy,
)
# EXCLUDED_SOURCES and SEARCHED_AND_EMPTY are deliberately NOT hoisted here.
# Several modules in this package define their own, and a bare name in the
# package namespace would silently be one of them. Reach them through
# `human.d2_in_vitro`.
from .d2_in_vitro import (
    EFFICACY_SCALE,
    HALOPERIDOL_HUMAN_D2_BINDING,
    HALOPERIDOL_HUMAN_D2_FUNCTIONAL,
    HALOPERIDOL_HUMAN_D2_INTRINSIC_EFFICACY,
    INVERSE_AGONISM_ABSTRACT_ONLY_SUPPORT,
    IntrinsicEfficacy,
    InVitroMeasurement,
    REGISTRY_EFFICACY_LABEL,
    REGISTRY_KI_LABEL,
    binding_ki_fold_spread,
    binding_ki_range_nM,
    efficacy_range,
    haloperidol_d2_binding,
    in_vitro_in_vivo_ratio_range,
    in_vitro_in_vivo_ratios,
    ratio_straddles_unity,
    registry_efficacy,
    registry_ki,
)
from .pet_observations import (
    HALOPERIDOL_D2_DATASET_IDS,
    HALOPERIDOL_D2_REPORTED_RELATIONS,
    ReportedOccupancyRelation,
    haloperidol_d2_manifests,
    haloperidol_d2_observations,
    occupancy_observations,
)
from .pet_validation import (
    PETOccupancyComparison,
    PETOccupancyObservation,
    compare_pet_occupancy,
    ng_per_mL_to_nM,
    observations_by_tracer,
    require_single_tracer,
)
from .heldout_validation import HeldOutHumanDrugResult, summarize_heldout
from .occupancy_validation import (
    CLINICAL_OCCUPANCY_THRESHOLDS,
    CLINICAL_WINDOW_PROVENANCE,
    OCCUPANCY_IS_NOT_THE_OUTCOME,
    PRE_REGISTERED_SPLITS,
    ClinicalOccupancyThreshold,
    ContainmentResult,
    HeldOutFold,
    HeldOutPrediction,
    HeldOutReport,
    SplitRule,
    SteadyStateOccupancy,
    as_heldout_results,
    basis_observations,
    build_folds,
    chain_testable_observations,
    clinical_window_check,
    dose_reaching_occupancy,
    kapur1996_containment,
    run_all_splits,
    run_split,
    split_rule,
    steady_state_occupancy,
)
from .failure_analysis import HUMAN_LAYER_FINDINGS, LayerFinding, blocking_findings, human_failure_analysis
from .gabaa_effect import GABAAOperationalModel, HUMAN_GABAA_OPERATIONAL_PRIOR
from .d2_effect import (
    D2Effect,
    D2FunctionalClass,
    PartialAgonistTransductionError,
    d2_effect_for_drug,
    d2_effect_from_occupancy,
    d2_functional_class,
    registered_d2_functional_classes,
)
from .dataset_manifest import HumanDatasetManifest, validate_manifest, validate_manifest_set
from .dmri_fallback import (
    DMRIFallback,
    TRACTOGRAPHY_LIMITATIONS,
)
from .ei_time_constants import (
    EXCITATORY_KEY,
    INHIBITORY_KEY,
    HumanEITimeConstants,
    from_delay_maps,
)
from .directed_prior import (
    BEST_HOMOLOGY_CASES,
    HOMOLOGY_LIMITATIONS,
    JIRSA_RETRY_STATUS,
    MacaqueDirectedPrior,
    DirectedPriorEdge,
    prior_from_weight_mapping,
)
from .provenance import (
    ABSENT,
    INFERRED,
    LEVELS as PROVENANCE_LEVELS,
    MEASURED,
    ProvenanceError,
    ProvenanceMap,
)
from .maps import (
    MISSING_POLICIES,
    HumanCoupling,
    unmeasured_regions,
    whole_brain_coupling,
    whole_brain_nodes,
)
from .scenario import (
    MECHANISM_GAIN_NOTE,
    HumanScenario,
    HumanScenarioReport,
    whole_brain_simulation,
)
from .neurotransmitter import (
    CITATIONS,
    RegionSignRule,
    SIGN_RULES,
    SignAssignment,
    SignRules,
)
from .fetch import (
    DEFAULT_CACHE_DIR,
    ChecksumMismatch,
    MissingCachedFile,
    cache_path,
    checksum_lines,
    download,
    ensure,
    fetch_all,
    is_cached,
    sha256_of,
    verify,
)
from .matrix import (
    HumanStructuralConnectome,
    load_connectome,
    read_connectome,
)
from .expression import (
    ABSENT,
    ASSUMPTION,
    HumanGeneExpression,
    HumanReceptorDensityMap,
    HumanReceptorGene,
    HUMAN_RECEPTOR_GENES,
    NORMALIZATIONS,
    REGIONS_WITHOUT_SAMPLES,
    STABLE_SYMBOLS,
    UNTHRESHOLDED_ONLY,
    availability,
    load_expression,
    read_expression,
    relative_density,
)

__all__ = [
    "atlas",
    "build",
    "directed_prior",
    "pk_parameters",
    "kp_uu",
    "KpUURecord",
    "HUMAN_KP_UU_RECORDS",
    "kp_uu_records",
    "transfer_kp_uu",
    "pk_validation",
    "ValidationOutcome",
    "HUMAN_PK_VALIDATION",
    "KNOWN_FAILURE_MODES",
    "validation_outcomes",
    "known_failure_mode",
    "receptor_parameters",
    "HumanReceptorParameter",
    "HUMAN_RECEPTOR_PARAMETERS",
    "human_receptor_parameters",
    "require_human_parameter",
    "GABAAOperationalModel",
    "HUMAN_GABAA_OPERATIONAL_PRIOR",
    "d2_effect",
    "D2Effect",
    "D2FunctionalClass",
    "PartialAgonistTransductionError",
    "d2_effect_for_drug",
    "d2_effect_from_occupancy",
    "d2_functional_class",
    "registered_d2_functional_classes",
    "HumanDatasetManifest",
    "validate_manifest",
    "validate_manifest_set",
    "pet_validation",
    "PETOccupancyComparison",
    "PETOccupancyObservation",
    "compare_pet_occupancy",
    "ng_per_mL_to_nM",
    "observations_by_tracer",
    "require_single_tracer",
    "occupancy_validation",
    "CLINICAL_OCCUPANCY_THRESHOLDS",
    "CLINICAL_WINDOW_PROVENANCE",
    "OCCUPANCY_IS_NOT_THE_OUTCOME",
    "PRE_REGISTERED_SPLITS",
    "ClinicalOccupancyThreshold",
    "ContainmentResult",
    "HeldOutFold",
    "HeldOutPrediction",
    "HeldOutReport",
    "SplitRule",
    "SteadyStateOccupancy",
    "as_heldout_results",
    "basis_observations",
    "build_folds",
    "chain_testable_observations",
    "clinical_window_check",
    "dose_reaching_occupancy",
    "kapur1996_containment",
    "run_all_splits",
    "run_split",
    "split_rule",
    "steady_state_occupancy",
    "dose_to_occupancy",
    "ChainLayer",
    "OccupancyPrediction",
    "PROVENANCE_LEVELS",
    "WeightScaling",
    "default_times_min",
    "predict_occupancy",
    "d2_ec50",
    "D2_PLASMA_EC50",
    "D2PlasmaEC50",
    "EC50Comparison",
    "EC50Fit",
    "EC50_PROVENANCE_CLASSES",
    "EMAX_CONVENTIONS",
    "HALOPERIDOL_D2_PLASMA_EC50",
    "HALOPERIDOL_D2_PLASMA_EC50_nM",
    "HALOPERIDOL_PLASMA_FREE_FRACTION",
    "PLASMA_BASES",
    "compare_with_reported_relations",
    "d2_plasma_ec50",
    "ec50_landscape",
    "fit_plasma_ec50",
    "registered_d2_drugs",
    "d2_in_vitro",
    "EFFICACY_SCALE",
    "HALOPERIDOL_HUMAN_D2_BINDING",
    "HALOPERIDOL_HUMAN_D2_FUNCTIONAL",
    "HALOPERIDOL_HUMAN_D2_INTRINSIC_EFFICACY",
    "INVERSE_AGONISM_ABSTRACT_ONLY_SUPPORT",
    "IntrinsicEfficacy",
    "InVitroMeasurement",
    "REGISTRY_EFFICACY_LABEL",
    "REGISTRY_KI_LABEL",
    "binding_ki_fold_spread",
    "binding_ki_range_nM",
    "efficacy_range",
    "haloperidol_d2_binding",
    "in_vitro_in_vivo_ratio_range",
    "in_vitro_in_vivo_ratios",
    "ratio_straddles_unity",
    "registry_efficacy",
    "registry_ki",
    "pet_observations",
    "HALOPERIDOL_D2_DATASET_IDS",
    "HALOPERIDOL_D2_REPORTED_RELATIONS",
    "ReportedOccupancyRelation",
    "haloperidol_d2_manifests",
    "haloperidol_d2_observations",
    "occupancy_observations",
    "HumanPKRecord",
    "HUMAN_PK_RECORDS",
    "human_pk_records",
    "dmri_fallback",
    "DMRIFallback",
    "TRACTOGRAPHY_LIMITATIONS",
    "ei_time_constants",
    "EXCITATORY_KEY",
    "INHIBITORY_KEY",
    "HumanEITimeConstants",
    "from_delay_maps",
    "DirectedPriorEdge",
    "MacaqueDirectedPrior",
    "prior_from_weight_mapping",
    "HOMOLOGY_LIMITATIONS",
    "BEST_HOMOLOGY_CASES",
    "JIRSA_RETRY_STATUS",
    "expression",
    "fetch",
    "fixtures",
    "maps",
    "matrix",
    "neurotransmitter",
    "scenario",
    "sources",
    "CITATIONS",
    "RegionSignRule",
    "SIGN_RULES",
    "SignAssignment",
    "SignRules",
    "HumanAtlas",
    "Region",
    "atlas_from_labels",
    "parse_label",
    "BuildReport",
    "HumanCircuit",
    "build_circuit_network",
    "MISSING_POLICIES",
    "HumanCoupling",
    "unmeasured_regions",
    "whole_brain_coupling",
    "whole_brain_nodes",
    "MECHANISM_GAIN_NOTE",
    "HumanScenario",
    "HumanScenarioReport",
    "whole_brain_simulation",
    "DEFAULT_CACHE_DIR",
    "ChecksumMismatch",
    "MissingCachedFile",
    "cache_path",
    "checksum_lines",
    "download",
    "ensure",
    "fetch_all",
    "is_cached",
    "sha256_of",
    "verify",
    "HumanStructuralConnectome",
    "load_connectome",
    "read_connectome",
    "ABSENT",
    "ASSUMPTION",
    "HumanGeneExpression",
    "HumanReceptorDensityMap",
    "HumanReceptorGene",
    "HUMAN_RECEPTOR_GENES",
    "NORMALIZATIONS",
    "REGIONS_WITHOUT_SAMPLES",
    "STABLE_SYMBOLS",
    "UNTHRESHOLDED_ONLY",
    "availability",
    "load_expression",
    "read_expression",
    "relative_density",
]
