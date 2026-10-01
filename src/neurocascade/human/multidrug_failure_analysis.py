"""Scope and failure analysis for the multi-drug chain (P22.5).

The P19.7 standard (``docs/p19-7-scope-and-failure-analysis.md``) applied to the
Phase 21-22 chain: state what was tested, attribute each failure to a layer, and
name the data that would repair it. Pre-registration commitment C4 binds this
task to regenerate every number from code, so nothing in here is a transcribed
result. Each figure is recomputed by calling the module that produced it
(``fda_popk``, ``fda_pk_bridge``, ``active_metabolites``,
``label_dose_result``, ``leave_one_drug_out``, ``clozapine_falsification``,
``occupancy_validation``). ``docs/p22-5-multidrug-failure-analysis.md`` embeds
:func:`layer_grid_markdown` and :func:`failure_matrix_markdown` verbatim, and a
test fails if the two drift apart.

This module adds no constant, no drug and no prediction. The only arithmetic
it does itself is to rearrange the existing outputs:

* the EC50 a single ziprasidone observation implies under Emax 1;
* the EC50 scaling band inside which a P22.2 pass would still pass. Dose scales
  exactly with EC50 in this chain, so the band is closed-form;
* aripiprazole's worst-case metabolite error at the two exposure ratios the
  sources disagree on (V7). This is *conditional on the label's equipotency
  assertion*, which P21.4 deliberately did not store as an affinity ratio, and
  it is labelled that way wherever it is printed.
"""
import functools
import math
from dataclasses import dataclass

from .active_metabolites import (
    ACTIVE_METABOLITES,
    GRUNDER_2008_ARIPIPRAZOLE_EC50_NG_PER_ML,
    equipotency_check,
    exposure_ratio_from_ec50_bases,
    require_parent_only_path,
)
from .clozapine_falsification import (
    CLOZAPINE_D2_EC50_PRIMARY,
    CLOZAPINE_D2_EC50_SENSITIVITY,
    clozapine_falsification,
    sensitivity_results,
)
from .d2_ec50 import D2_PLASMA_EC50
from .d2_effect import d2_functional_class, registered_d2_functional_classes
from .dose_to_occupancy import _require_pk_record
from .fda_pk_bridge import refusals
from .fda_popk import FDA_POPPK_RECORDS, half_life_comparisons
from .ke0 import HUMAN_KE0_RECORDS
from .kp_uu import kp_uu_records
from .label_dose_prediction import DENOMINATOR as LABEL_DENOMINATOR
from .label_dose_result import LABEL_TRANSCRIPTIONS, label_dose_summary, label_dose_table
from .leave_one_drug_out import (
    DENOMINATOR as CENSUS,
    HELD_OUT_EC50,
    OBSERVATIONS,
    SEARCHED_AND_EMPTY,
    leave_one_drug_out_table,
)
from .occupancy_validation import (
    CLINICAL_OCCUPANCY_THRESHOLDS,
    chain_testable_observations,
    run_split,
)

#: The pass criterion of P22.2, fixed in P22.1: twofold on the midpoint.
_TWOFOLD = math.log10(2.0)


@functools.lru_cache(maxsize=1)
def _label_table():
    """P22.2's table, computed once: each row integrates the chain to steady
    state, and both the grid and the matrix read it."""
    return label_dose_table()


# ---------------------------------------------------------------------------
# Per-drug status of every layer
# ---------------------------------------------------------------------------


def _first_line(text):
    return str(text).split(";")[0]


def _pk_status(drug):
    try:
        # Frozen at the P22 sources: this result was scored against the PK
        # registry as it stood then, before P25.2 added literature records.
        _require_pk_record(drug, "PO", None, include_literature=False)
    except ValueError:
        rows = [r for r in FDA_POPPK_RECORDS if r.drug == drug and r.route == "PO"]
        if not rows:
            return "refused: no oral popPK row"
        reasons = [(refusals(r) or ("bridged",))[0] for r in rows]
        if any("barrier" in reason for reason in reasons):
            barrier = [reason.split("barrier: ")[1].rstrip(")") for reason in reasons if "barrier" in reason]
            return "refused: oral row %s" % "/".join(sorted(set(barrier)))
        if any("analyte" in reason for reason in reasons):
            return "refused: oral row carries metabolite"
        return "refused: %s" % _first_line(reasons[0])
    return "accepted"


def _analyte_status(drug):
    try:
        require_parent_only_path(drug, target="D2")
    except ValueError:
        return "refused"
    if drug in ACTIVE_METABOLITES:
        return "bounded (%.4f)" % ACTIVE_METABOLITES[drug][0].max_occupancy_error()
    return "open"


def _occupancy_status(drug):
    if drug in D2_PLASMA_EC50:
        return "chain registry"
    if drug in HELD_OUT_EC50:
        return "P22.3 module"
    if drug == CLOZAPINE_D2_EC50_PRIMARY.drug:
        return "P22.4 module"
    return "none held"


def _observation_status(drug, result):
    if result.status == "searched-and-empty":
        return "searched-and-empty (%d sources)" % len(SEARCHED_AND_EMPTY[drug].searched)
    return "n=%d, %d groups" % (result.n, result.n_independent_groups)


def _label_status(drug, rows):
    if drug not in LABEL_DENOMINATOR:
        return "not in denominator"
    row = rows[drug]
    if row.label is None:
        return "not transcribed (refused)"
    return "scored"


def _outcome(drug, label_rows, lodo):
    parts = []
    if drug in label_rows and label_rows[drug].score.refusal is None:
        score = label_rows[drug].score
        parts.append("V8 %s e=%+.4f" % ("pass" if score.passed else "fail", score.e))
    if lodo.status == "scored":
        parts.append("P22.3 %s MAE %.4f" % ("pass" if lodo.passed else "fail", lodo.mae))
    if drug == CLOZAPINE_D2_EC50_PRIMARY.drug:
        parts.append("V11 Outcome %s" % clozapine_falsification().outcome)
    return "; ".join(parts) or "no comparison"


@dataclass(frozen=True)
class DrugLayerRow:
    drug: str
    analyte: str
    pk: str
    exposure: str
    occupancy_constant: str
    heldout_observations: str
    label: str
    outcome: str


def layer_grid():
    """One row per drug of the nine-drug P22.3 census, every layer's status."""
    label_rows = {row.drug: row for row in _label_table()}
    lodo = {result.drug: result for result in leave_one_drug_out_table()}
    rows = []
    for drug in CENSUS:
        rows.append(
            DrugLayerRow(
                drug=drug,
                analyte=_analyte_status(drug),
                pk=_pk_status(drug),
                exposure="not-required record" if kp_uu_records(drug) else "no record",
                occupancy_constant=_occupancy_status(drug),
                heldout_observations=_observation_status(drug, lodo[drug]),
                label=_label_status(drug, label_rows),
                outcome=_outcome(drug, label_rows, lodo[drug]),
            )
        )
    return tuple(rows)


def layer_grid_markdown():
    header = (
        "| drug | analyte | pk (oral) | exposure (Kp,uu) | occupancy constant "
        "| held-out PET obs | label | Phase 22 outcome |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for r in layer_grid():
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s |"
            % (r.drug, r.analyte, r.pk, r.exposure, r.occupancy_constant,
               r.heldout_observations, r.label, r.outcome)
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The regenerated numbers
# ---------------------------------------------------------------------------


def _ec50_pass_band(window, reading):
    """EC50 multipliers k for which the scaled window still passes P22.2.

    D65 and D78 both scale by k, so e shifts by log10 k, and overlap holds while
    k*D65 <= L_high and k*D78 >= L_low.
    """
    d65, d78 = window.d65_mg_per_day, window.d78_mg_per_day
    e = math.log10(math.sqrt(d65 * d78) / math.sqrt(reading.l_low * reading.l_high))
    low = max(10 ** (-_TWOFOLD - e), reading.l_low / d78)
    high = min(10 ** (_TWOFOLD - e), reading.l_high / d65)
    return low, high


def _chain_testable(drug):
    """Observations pairing an oral dose with an occupancy, or ``None`` when the
    drug has no registered PET observation set to look in at all."""
    try:
        return len(chain_testable_observations(drug))
    except ValueError:
        return None


def _pk_numbers():
    first = [(refusals(r) or ("BRIDGED",))[0] for r in FDA_POPPK_RECORDS]
    by_kind = {
        "bridged": sum(1 for f in first if f == "BRIDGED"),
        "route": sum(1 for f in first if f.startswith("route")),
        "analyte": sum(1 for f in first if f.startswith("analyte")),
        "searched-and-empty": sum(1 for f in first if f.startswith("status")),
        "reference weight": sum(1 for f in first if "reference subject" in f),
    }
    comparisons = half_life_comparisons()

    def route_of(c):
        routes = {
            r.route for r in FDA_POPPK_RECORDS
            if r.application_number == c.application_number and r.analyte == c.analyte
        }
        (route,) = routes
        return route

    by_route = {}
    for c in comparisons:
        n, ok = by_route.get(route_of(c), (0, 0))
        by_route[route_of(c)] = (n + 1, ok + int(c.agrees_within_twofold))
    return {
        "fda_rows": len(FDA_POPPK_RECORDS),
        "fda_first_refusal": by_kind,
        "chain_pk_drugs": tuple(d for d in CENSUS if _pk_status(d).startswith("accepted")),
        "dose_occupancy_pairs": {
            d: _chain_testable(d) for d in CENSUS if _pk_status(d).startswith("accepted")
        },
        "half_life_n": len(comparisons),
        "half_life_within_twofold": sum(c.agrees_within_twofold for c in comparisons),
        "half_life_by_route": by_route,
        "half_life_worst_fold": max(c.fold_error for c in comparisons),
    }


def max_occupancy_error(drive_ratio):
    """``d / (1 + sqrt(1 + d))**2``, the bound
    :meth:`ActiveMetabolite.max_occupancy_error` computes, for a drive ratio the
    record does not store."""
    return drive_ratio / (1.0 + math.sqrt(1.0 + drive_ratio)) ** 2


def _analyte_numbers():
    risperidone = ACTIVE_METABOLITES["risperidone"][0]
    haloperidol = ACTIVE_METABOLITES["haloperidol"][0]
    aripiprazole = ACTIVE_METABOLITES["aripiprazole"][0]
    derived = {
        region: exposure_ratio_from_ec50_bases(*pair)
        for region, pair in GRUNDER_2008_ARIPIPRAZOLE_EC50_NG_PER_ML.items()
    }
    check = equipotency_check()
    return {
        "risperidone_drive_ratio": risperidone.drive_ratio(),
        "risperidone_max_error": risperidone.max_occupancy_error(),
        "haloperidol_drive_ratio": haloperidol.drive_ratio(),
        "haloperidol_max_error": haloperidol.max_occupancy_error(),
        "haloperidol_affinity_species": haloperidol.affinity_species,
        "aripiprazole_label_ratio": aripiprazole.exposure_ratio,
        "aripiprazole_derived_ratio": derived,
        "aripiprazole_ratio_fold": derived["putamen"] / aripiprazole.exposure_ratio,
        # Conditional on the label's equipotency assertion (affinity ratio 1),
        # which P21.4 kept out of ``affinity_ratio`` on purpose; so d = r.
        "aripiprazole_max_error_label_equipotent": max_occupancy_error(aripiprazole.exposure_ratio),
        "aripiprazole_max_error_derived_equipotent": max_occupancy_error(derived["putamen"]),
        "v6_predicted": check.predicted_ng_per_mL,
        "v6_measured": check.measured_ng_per_mL,
        "v6_fold": check.fold_error,
    }


def _occupancy_numbers():
    lodo = {r.drug: r for r in leave_one_drug_out_table()}
    zip_obs = next(o for o in OBSERVATIONS if o.drug == "ziprasidone")
    zip_row = HELD_OUT_EC50["ziprasidone"]
    implied = zip_obs.plasma_ng_per_mL * (1.0 - zip_obs.observed_occupancy) / zip_obs.observed_occupancy
    zip_ec50_ng = zip_row.ec50_nM * zip_row.molar_mass_g_per_mol / 1000.0
    split_b = run_split("B-leave-one-study-out").mean_absolute_error
    split_d = run_split("D-external-parameter")
    clozapine = sensitivity_results()
    threshold_tracer = CLINICAL_OCCUPANCY_THRESHOLDS[0].tracer
    same_tracer = [r for r in clozapine if _tracer_of(r) == threshold_tracer]
    constants = (
        tuple(D2_PLASMA_EC50.values())
        + tuple(HELD_OUT_EC50.values())
        + (CLOZAPINE_D2_EC50_PRIMARY,)
        + CLOZAPINE_D2_EC50_SENSITIVITY
    )
    uchida = [r for r in CLOZAPINE_D2_EC50_SENSITIVITY if r.tracer == threshold_tracer]
    return {
        "split_b_mae": split_b,
        "split_d_mae": split_d.mean_absolute_error,
        "split_d_bias": split_d.bias,
        "aripiprazole_mae": lodo["aripiprazole"].mae,
        "aripiprazole_bias": lodo["aripiprazole"].bias,
        "aripiprazole_n": lodo["aripiprazole"].n,
        "aripiprazole_groups": lodo["aripiprazole"].n_independent_groups,
        "ziprasidone_error": lodo["ziprasidone"].bias,
        "ziprasidone_ec50_ng": zip_ec50_ng,
        "ziprasidone_implied_ec50_ng": implied,
        "ziprasidone_implied_fold": zip_ec50_ng / implied,
        "ziprasidone_ec50_regimen": zip_row.regimen,
        "constants_total": len(constants),
        "constants_region_none": tuple(r.drug for r in constants if r.region is None),
        "constants_tracer_none": tuple(r.drug for r in constants if r.tracer is None),
        "threshold_tracer": threshold_tracer,
        "clozapine_primary_tracer": CLOZAPINE_D2_EC50_PRIMARY.tracer,
        "clozapine_emax_convention_spread": max(r.ec50_ng_per_mL for r in uchida)
        / min(r.ec50_ng_per_mL for r in uchida),
        "clozapine_same_tracer": tuple(
            (r.ec50_ng_per_mL, r.emax, r.c65_ng_per_mL, r.fold_c65_over_effective_high, r.outcome)
            for r in same_tracer
        ),
    }


def _tracer_of(result):
    for record in (CLOZAPINE_D2_EC50_PRIMARY,) + CLOZAPINE_D2_EC50_SENSITIVITY:
        if (record.region, record.ec50_ng_per_mL, record.emax) == (
            result.region, result.ec50_ng_per_mL, result.emax
        ):
            return record.tracer
    raise KeyError(result.region)


def _label_numbers():
    table = {row.drug: row for row in _label_table()}
    summary = label_dose_summary(_label_table())
    bands = {
        drug: _ec50_pass_band(table[drug].window, LABEL_TRANSCRIPTIONS[drug].scoring_reading)
        for drug in LABEL_TRANSCRIPTIONS
    }
    olanzapine = table["olanzapine"]
    return {
        "n_pass": summary["n_pass"],
        "n_scored": summary["n_scored"],
        "denominator": summary["denominator"],
        "e": {d: table[d].score.e for d in LABEL_TRANSCRIPTIONS},
        "windows": {
            d: (table[d].window.d65_mg_per_day, table[d].window.d78_mg_per_day)
            for d in LABEL_TRANSCRIPTIONS
        },
        "ec50_pass_band": bands,
        "first_refusal": {
            row.drug: row.score.refusal for row in table.values() if row.score.refusal
        },
        "olanzapine_d65_over_label_low": olanzapine.window.d65_mg_per_day
        / LABEL_TRANSCRIPTIONS["olanzapine"].scoring_reading.l_low,
        "olanzapine_label_low": LABEL_TRANSCRIPTIONS["olanzapine"].scoring_reading.l_low,
        "haloperidol_label_span": LABEL_TRANSCRIPTIONS["haloperidol"].scoring_reading.l_high
        / LABEL_TRANSCRIPTIONS["haloperidol"].scoring_reading.l_low,
    }


def _clinical_numbers():
    primary = clozapine_falsification()
    return {
        "clozapine_c65": primary.c65_ng_per_mL,
        "clozapine_c78": primary.c78_ng_per_mL,
        "clozapine_fold": primary.fold_c65_over_effective_high,
        "clozapine_effective_range": (primary.effective_low_ng_per_mL, primary.effective_high_ng_per_mL),
        "clozapine_occupancy_in_range": (
            primary.occupancy_at_effective_low,
            primary.occupancy_at_effective_high,
        ),
        "clozapine_outcomes": tuple(r.outcome for r in sensitivity_results()),
    }


def regenerated_numbers():
    """Every figure the P22.5 document prints, recomputed now."""
    lodo = leave_one_drug_out_table()
    return {
        "census": CENSUS,
        "pk": _pk_numbers(),
        "analyte": _analyte_numbers(),
        "occupancy": _occupancy_numbers(),
        "observations": {
            "searched_and_empty": tuple(r.drug for r in lodo if r.status == "searched-and-empty"),
            "sources_opened": sum(len(s.searched) for s in SEARCHED_AND_EMPTY.values()),
            "scored": tuple(r.drug for r in lodo if r.status == "scored"),
        },
        "label": _label_numbers(),
        "clinical": _clinical_numbers(),
        "no_kp_uu_record": tuple(d for d in CENSUS if not kp_uu_records(d)),
        "ke0_illustrative": sum(1 for k in HUMAN_KE0_RECORDS if k.status == "illustrative"),
        "partial_agonists": tuple(
            d for d in registered_d2_functional_classes()
            if d2_functional_class(d).kind == "partial_agonist"
        ),
    }


# ---------------------------------------------------------------------------
# The failure matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailureRow:
    layer: str
    drugs: tuple
    evidence: str
    attribution: str
    data_required: str

    def __post_init__(self):
        if not (self.layer and self.drugs and self.evidence and self.attribution and self.data_required):
            raise ValueError("every failure row needs a layer, drugs, evidence, attribution and a data requirement")


def _fmt_drugs(drugs):
    return ", ".join(drugs)


def failure_matrix(numbers=None):
    n = regenerated_numbers() if numbers is None else numbers
    pk, an, oc, lb, cl = n["pk"], n["analyte"], n["occupancy"], n["label"], n["clinical"]
    no_pk = tuple(d for d in n["census"] if d not in pk["chain_pk_drugs"])
    no_kp = n["no_kp_uu_record"]
    routes = pk["half_life_by_route"]
    route_text = ", ".join("%s %d/%d" % (r, ok, total) for r, (total, ok) in sorted(routes.items()))
    refusal = pk["fda_first_refusal"]
    band = lb["ec50_pass_band"]
    same = oc["clozapine_same_tracer"]
    same_text = "; ".join(
        "EC50 %g Emax %.2f: %s" % (
            ec50, emax,
            "65% unreachable" if c65 is None else "C65 %.0f ng/mL, %.2fx the range top" % (c65, fold),
        )
        for ec50, emax, c65, fold, _ in same
    )
    return (
        FailureRow(
            layer="pk: oral dose to plasma",
            drugs=no_pk,
            evidence=(
                "%d of %d census drugs have no oral PK record the chain accepts. Of %d FDA "
                "population-PK rows, %d bridges; first refusal: %d route (depot/IV), "
                "%d metabolite-carrying analyte, %d searched-and-empty, %d unprinted "
                "reference weight. For the drivable drugs, observations pairing an oral dose "
                "with an occupancy: %s, so the arm is untested against occupancy"
                % (len(no_pk), len(n["census"]), pk["fda_rows"], refusal["bridged"],
                   refusal["route"], refusal["analyte"], refusal["searched-and-empty"],
                   refusal["reference weight"],
                   ", ".join("%s %s" % (d, "no PET set" if k is None else k)
                             for d, k in pk["dose_occupancy_pairs"].items()))
            ),
            attribution=(
                "source limitation, not a model failure: population PK entered this class "
                "with the depots, and the oral originals were never fitted or are redacted. "
                "P21.3 searched FDA reviews only; the journal literature was not searched"
            ),
            data_required=(
                "per drug, an adult oral popPK with typical values, units, reference weight "
                "and a parent-only analyte printed; the first task is a journal search for "
                "these, which no task here has run"
            ),
        ),
        FailureRow(
            layer="pk: depot disposition (V5)",
            drugs=("risperidone", "paliperidone", "aripiprazole", "olanzapine", "amisulpride"),
            evidence=(
                "%d of %d FDA-implied half-lives within twofold of the label; by route %s; "
                "worst %.1fx"
                % (pk["half_life_within_twofold"], pk["half_life_n"], route_text,
                   pk["half_life_worst_fold"])
            ),
            attribution=(
                "the source fits, not the chain: depot release parked in a peripheral "
                "compartment makes the slowest eigenvalue a formulation constant. No driven "
                "prediction is affected, because the chain refuses every depot row"
            ),
            data_required=(
                "depot fits that model release as an explicit input (the RELPREVV pattern), "
                "or oral immediate-release fits for the same molecules"
            ),
        ),
        FailureRow(
            layer="analyte: active metabolite",
            drugs=("risperidone", "aripiprazole", "paliperidone"),
            evidence=(
                "risperidone d = %.1f, worst-case error %.3f, %.0fx the split-B MAE %.4f; "
                "aripiprazole exposure ratio label %.2f vs Gruender-derived %.2f (%.1fx, V7), "
                "worst-case error %.4f vs %.4f conditional on label equipotency; equipotency "
                "check predicts %.3f vs measured %.2f ng/mL (%.2fx, V6)"
                % (an["risperidone_drive_ratio"], an["risperidone_max_error"],
                   an["risperidone_max_error"] / oc["split_b_mae"], oc["split_b_mae"],
                   an["aripiprazole_label_ratio"], an["aripiprazole_derived_ratio"]["putamen"],
                   an["aripiprazole_ratio_fold"], an["aripiprazole_max_error_label_equipotent"],
                   an["aripiprazole_max_error_derived_equipotent"], an["v6_predicted"],
                   an["v6_measured"], an["v6_fold"])
            ),
            attribution=(
                "risperidone: ill-posed on the parent axis, not unfitted. Aripiprazole: the "
                "two sources disagree on the exposure ratio and no metabolite constant exists. "
                "Haloperidol is bounded at %.4f on a %s affinity and passes the gate"
                % (an["haloperidol_max_error"], an["haloperidol_affinity_species"])
            ),
            data_required=(
                "risperidone: an oral active-moiety-sum popPK to pair with the moiety EC50 "
                "(nothing on the parent axis ever). Aripiprazole: an in-vivo human EC50 for "
                "dehydro-aripiprazole alone, or a steady-state oral PET printing both bases "
                "and the metabolite ratio in the same subjects"
            ),
        ),
        FailureRow(
            layer="exposure: Kp,uu record",
            drugs=no_kp,
            evidence=(
                "%d of %d census drugs carry no Kp,uu record, and the chain's "
                "double-counting guard refuses a drug without one"
                % (len(no_kp), len(n["census"]))
            ),
            attribution=(
                "a structural gate, not a failure: it is masked today because every drug "
                "it would stop is already refused at pk"
            ),
            data_required=(
                "no new measurement: a 'not-required' record stating the EC50 was fitted "
                "against plasma in living people, entered with the drug's EC50"
            ),
        ),
        FailureRow(
            layer="occupancy constant: between-study spread",
            drugs=("ziprasidone", "haloperidol"),
            evidence=(
                "ziprasidone error %+.4f (n=1); the point implies EC50 %.1f ng/mL, %.1fx below "
                "the %.1f used. Haloperidol with published constants: MAE %.4f, bias %+.4f "
                "(split D). Aripiprazole, the regimen-matched case, MAE %.4f over %d groups"
                % (oc["ziprasidone_error"], oc["ziprasidone_implied_ec50_ng"],
                   oc["ziprasidone_implied_fold"], oc["ziprasidone_ec50_ng"],
                   oc["split_d_mae"], oc["split_d_bias"], oc["aripiprazole_mae"],
                   oc["aripiprazole_groups"])
            ),
            attribution=(
                "the occupancy layer, as a disagreement between published constants rather "
                "than a wrong functional form: the hyperbola passes where regimen and tracer "
                "match (V9) and misses where they do not"
            ),
            data_required=(
                "per-subject paired plasma and occupancy, tracer and region printed, disjoint "
                "from the EC50 source and regimen-matched to it; for ziprasidone, a "
                "steady-state set to set against Suzuki 2013"
            ),
        ),
        FailureRow(
            layer="occupancy constant: provenance",
            drugs=tuple(sorted(set(oc["constants_region_none"] + oc["constants_tracer_none"]))) + ("clozapine",),
            evidence=(
                "of %d D2 EC50 rows used anywhere in Phase 22, region unprinted for %s, "
                "tracer unprinted for %s. Clozapine's primary constant is %s against "
                "thresholds measured with %s"
                % (oc["constants_total"], _fmt_drugs(oc["constants_region_none"]),
                   _fmt_drugs(oc["constants_tracer_none"]), oc["clozapine_primary_tracer"],
                   oc["threshold_tracer"])
            ),
            attribution=(
                "source reporting, inherited by the layer. The 2-2.5x tracer floor (P19.7) "
                "is wider than both P22.2 pass bands, so a pass cannot separate EC50 "
                "candidates inside it"
            ),
            data_required=(
                "constants fitted with region and tracer printed; for clozapine a striatal "
                "[11C]raclopride plasma EC50 read first-hand, since both raclopride rows are "
                "second-hand"
            ),
        ),
        FailureRow(
            layer="regimen",
            drugs=("ziprasidone", "aripiprazole", "haloperidol"),
            evidence=(
                "ziprasidone EC50 fitted on '%s', tested on one single dose; aripiprazole's "
                "only EC50 is single-dose against a steady-state label ratio; haloperidol's "
                "fitted on depot and driven oral; every P22.2 window assumes once-daily dosing"
                % oc["ziprasidone_ec50_regimen"]
            ),
            attribution=(
                "a mismatch between the fitting and the driving regimen, which the chain "
                "carries as text and cannot correct: it has no accumulation-dependent term "
                "beyond PK superposition"
            ),
            data_required="oral steady-state PET with paired plasma, per drug, at the dosing frequency the label uses",
        ),
        FailureRow(
            layer="effect-site kinetics",
            drugs=("quetiapine",),
            evidence=(
                "no ke0 in the chain; all %d human ke0 records are illustrative. Quetiapine's "
                "Nord 2011 constant exists but absorbs peak-trough hysteresis, so it is not "
                "ingested"
                % n["ke0_illustrative"]
            ),
            attribution="structural: a direct-effect hyperbola cannot represent occupancy that lags plasma",
            data_required="time-resolved PET at peak and trough with paired plasma, enough to fit a human ke0 per drug",
        ),
        FailureRow(
            layer="transduction",
            drugs=n["partial_agonists"],
            evidence="occupancy-equals-effect refused for %s (partial agonist)" % _fmt_drugs(n["partial_agonists"]),
            attribution="structural, and searched-and-empty: no single in-vivo human intrinsic activity exists",
            data_required="none exists to obtain (P21.5); a different transducer form, not a number, would be needed",
        ),
        FailureRow(
            layer="occupancy to clinical outcome",
            drugs=("clozapine", "olanzapine"),
            evidence=(
                "clozapine C65 %.0f ng/mL, %.2fx the top of the %g-%g ng/mL effective range, "
                "which the chain maps to %.3f-%.3f occupancy; outcomes across candidates %s. "
                "Same-tracer candidates only: %s. Olanzapine D65 sits at %.2fx the label's "
                "%g mg/day target"
                % (cl["clozapine_c65"], cl["clozapine_fold"],
                   cl["clozapine_effective_range"][0], cl["clozapine_effective_range"][1],
                   cl["clozapine_occupancy_in_range"][0], cl["clozapine_occupancy_in_range"][1],
                   "".join(cl["clozapine_outcomes"]), same_text,
                   lb["olanzapine_d65_over_label_low"], lb["olanzapine_label_low"])
            ),
            attribution=(
                "the hypothesis, not the chain: the occupancy prediction is right and the "
                "occupancy-threshold-as-sole-mediator reading is falsified for clozapine"
            ),
            data_required=(
                "per-drug clinical response against something other than D2 occupancy: the "
                "off-target fingerprint data Phase 23 names"
            ),
        ),
        FailureRow(
            layer="ground truth: held-out PET observations",
            drugs=n["observations"]["searched_and_empty"],
            evidence=(
                "%d of %d drugs searched-and-empty, %d sources opened and recorded"
                % (len(n["observations"]["searched_and_empty"]), len(n["census"]),
                   n["observations"]["sources_opened"])
            ),
            attribution="the test side, not the model: the paired values exist in closed or unpaired tables",
            data_required=(
                "the closed tables named in SEARCHED_AND_EMPTY (e.g. Nordstrom 1995, Kapur "
                "1999, Kessler 2006 for clozapine), read first-hand with same-visit pairing"
            ),
        ),
        FailureRow(
            layer="ground truth: label dose ranges",
            drugs=tuple(LABEL_TRANSCRIPTIONS),
            evidence=(
                "%d pass / %d scored / %d; olanzapine passes for EC50 x%.2f-%.2f, haloperidol "
                "for x%.2f-%.2f; haloperidol's scored label range spans %.0f-fold"
                % (lb["n_pass"], lb["n_scored"], lb["denominator"],
                   band["olanzapine"][0], band["olanzapine"][1],
                   band["haloperidol"][0], band["haloperidol"][1],
                   lb["haloperidol_label_span"])
            ),
            attribution=(
                "low power, not a failure: the ground truth is wider than the tracer floor, "
                "so the passes cannot discriminate"
            ),
            data_required=(
                "a ground-truth dose quantity narrower than twofold per drug, from fixed-dose "
                "trial arms rather than label ranges"
            ),
        ),
    )


def failure_matrix_markdown():
    lines = [
        "| layer | drugs | evidence (regenerated) | attribution | data that would repair it |",
        "|---|---|---|---|---|",
    ]
    for row in failure_matrix():
        lines.append(
            "| %s | %s | %s | %s | %s |"
            % (row.layer, _fmt_drugs(row.drugs), row.evidence, row.attribution, row.data_required)
        )
    return "\n".join(lines)
