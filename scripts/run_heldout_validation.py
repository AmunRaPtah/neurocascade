"""P8.2: run the held-out validation and write the results.

    PYTHONPATH=src python scripts/run_heldout_validation.py

Writes, under `results/heldout/`:

    heldout_validation.txt   the whole report
    heldout_coverage.csv     one row per compound per chain link
    heldout_errors.csv       one row per compound per endpoint per dose
    heldout_reachable.png    what the model can and cannot reach
    heldout_coverage.png     the coverage matrix
    heldout_tone_ceiling.png what an antagonist can remove at each target

Everything printed is computed here. No number in the report is transcribed
from the prose of `docs/architecture.md`, so a claim that stops being true
stops being printed.
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neurocascade import fingerprint as fp                    # noqa: E402
from neurocascade import heldout as ho                        # noqa: E402
from neurocascade.calibration import endpoint_forward as efwd  # noqa: E402
from neurocascade.calibration import heldout_forward as hf     # noqa: E402
from neurocascade.calibration import heldout_validation as hv  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "heldout")

#: The thresholds the monotonicity turnover is tested at. Two, deliberately:
#: a turnover that survives 1 sd but not 1.96 is a weaker result than one that
#: survives both, and reporting only the threshold that passes would be
#: choosing the test after seeing the data.
SIGMA_LEVELS = (1.0, 1.96)


def _wrap(text, width=74, indent=""):
    words, lines, current = str(text).split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(indent + current)
            current = word
        else:
            current = (current + " " + word).strip()
    if current:
        lines.append(indent + current)
    return "\n".join(lines)


def _rule(title, char="="):
    return "\n%s\n%s" % (title, char * 72)


def coverage_section(out, compounds):
    out.append(_rule("1. Coverage: which links of the chain each compound has"))
    out.append(_wrap(
        "A dose-to-behaviour prediction needs all five links. This table is "
        "the acceptance criterion 'quantitative error per compound per "
        "endpoint' meeting reality: an error can only be computed where the "
        "chain closes."
    ))
    out.append("")
    header = "%-13s %s" % ("compound", "  ".join(
        "%-9s" % link[:9] for link in hv.CHAIN_LINKS
    ))
    out.append(header)
    out.append("-" * len(header))
    statuses = {}
    for name, compound in sorted(compounds.items()):
        status = hv.chain_status(compound)
        statuses[name] = status
        out.append("%-13s %s" % (name, "  ".join(
            "%-9s" % ("yes" if status.links[link] else "NO")
            for link in hv.CHAIN_LINKS
        )))
    complete = [n for n, s in statuses.items() if s.complete]
    out.append("")
    out.append(_wrap(
        "Compounds with a complete chain: %d of %d%s"
        % (len(complete), len(statuses),
           (" (%s)" % ", ".join(complete)) if complete else "")
    ))
    out.append("")
    for name, status in sorted(statuses.items()):
        out.append("  %s" % name)
        for link in hv.CHAIN_LINKS:
            out.append(_wrap(
                "%-17s %s" % (link + ":", status.detail[link]),
                width=70, indent="      ",
            ))
        out.append("")
    return statuses


def binding_section(out, compounds):
    out.append(_rule("2. The binding fingerprints, and what they omit"))
    out.append(_wrap(
        "Every affinity is a published measurement with its own citation. "
        "'tested-inactive' counts targets a panel screened and found below "
        "its ceiling; 'never-tested' counts targets nobody measured. Only "
        "the first licenses calling a compound's model complete over that "
        "target."
    ))
    out.append("")
    header = ("%-13s %-9s %-9s %-11s %-11s  %s"
              % ("compound", "modelled", "censored", "inactive", "untested",
                 "targets"))
    out.append(header)
    out.append("-" * len(header))
    for name, compound in sorted(compounds.items()):
        fingerprint = compound.fingerprint()
        censored = sum(1 for r in compound.ki if r.is_censored)
        out.append("%-13s %-9d %-9d %-11d %-11d  %s" % (
            name, len(fingerprint.entries), censored,
            len(compound.tested_inactive), len(compound.never_tested),
            ", ".join(fingerprint.targets),
        ))
    out.append("")


def tone_section(out):
    out.append(_rule("3. The tone ceiling: what an antagonist can ever remove"))
    out.append(_wrap(
        "A competitive antagonist removes the activation the endogenous "
        "transmitter was supplying and no more, so its reach is bounded by "
        "P6.2's MEASURED tone before any dose, gain or density is chosen. "
        "Targets absent from this table have no measured basal transmitter "
        "in this repository at all, and an antagonist there changes nothing "
        "whatsoever."
    ))
    out.append("")
    ceiling = hv.tone_ceiling()
    out.append("%-14s %-12s %s" % ("target", "max |d_act|", "reading"))
    out.append("-" * 58)
    for target, value in sorted(ceiling.items(), key=lambda kv: -kv[1]):
        reading = ("substantial" if value > 0.1
                   else "marginal" if value > 0.01 else "negligible")
        out.append("%-14s %-12.5f %s" % (target, value, reading))
    without = sorted(
        t for t, spec in fp.TARGETS.items()
        if not spec.is_transporter and t not in ceiling
    )
    out.append("")
    out.append(_wrap("No measured tone (antagonist reach is exactly zero): %s"
                     % ", ".join(without)))
    out.append("")
    return ceiling


def _response_for(target, models, responses):
    if target not in responses:
        responses[target] = hf.sweep_target(models[target], target)
    return responses[target]


def reachable_section(out, models, responses, targets):
    out.append(_rule("4. What the model can reach, over ALL gains"))
    out.append(_wrap(
        "The endpoint index saturates, so there is a largest effect the "
        "model can produce at each endpoint for ANY gain whatsoever. An "
        "observation outside this interval is refuted by the model's "
        "structure rather than by a parameter choice. Convergence of each "
        "range is checked rather than assumed; 'drift' is how much the index "
        "still moved over the outermost decade of the sweep."
    ))
    out.append("")
    for target in targets:
        response = _response_for(target, models, responses)
        ranges = hf.reachable_index_range(response)
        out.append("  target %s (gene %s, %s)" % (
            target, fp.TARGETS[target].gene,
            "gain %s" % hf.gain_standing(target)[0],
        ))
        out.append("    %-22s %-9s %-9s %-10s %s"
                   % ("endpoint", "ratio lo", "ratio hi", "drift", "converged"))
        for key in hf.ENDPOINT_KEYS:
            reach = ranges[key]
            out.append("    %-22s %-9.4f %-9.4f %-10.2e %s" % (
                key, reach.ratio_lo, reach.ratio_hi, reach.drift,
                "yes" if reach.converged else "NO",
            ))
        out.append("")
    return responses


def direction_section(out, compounds, models, responses):
    out.append(_rule("5. Direction: a parameter-free sign prediction"))
    out.append(_wrap(
        "The sign of an endpoint's response is sign(gain) x "
        "sign(delta_activation) x the endpoint's own slope, and none of the "
        "three needs the gain's MAGNITUDE. This is the one behavioural "
        "prediction in this task that costs no fitted parameter."
    ))
    out.append("")
    header = ("%-12s %-20s %-12s %-11s %-11s %s"
              % ("compound", "endpoint", "via", "predicted", "observed",
                 "match"))
    out.append(header)
    out.append("-" * len(header))
    rows, scored, correct = [], 0, 0
    for name, compound in sorted(compounds.items()):
        actionable = [
            target for target in compound.modelled_targets
            if hv.behavioural_path_status_for(compound, target) == "available"
        ]
        for observation in compound.behavioural_observations:
            if not actionable:
                predicted, via, slope = "no-change", "no path", 0.0
            else:
                # One target per compound here. Where several are actionable
                # the gains are unidentifiable from a single observation
                # (P2.3's rank-one Fisher information, at a different scale),
                # so the report attributes the direction to the first and the
                # coverage table above names all of them.
                target = actionable[0]
                response = _response_for(target, models, responses)
                x_sign = hv.delta_activation_sign_for(compound, target) \
                    * hf.transduction_sign(target)[1]
                predicted, slope = hv.endpoint_direction(
                    response, observation.endpoint, x_sign
                )
                via = target
            match = "yes" if predicted == observation.direction else "NO"
            scored += 1
            correct += match == "yes"
            out.append("%-12s %-20s %-12s %-11s %-11s %s" % (
                name, observation.endpoint, via, predicted,
                observation.direction, match,
            ))
            rows.append({
                "compound": name, "endpoint": observation.endpoint,
                "dose_mg_per_kg": observation.dose_mg_per_kg,
                "via_target": via, "predicted_direction": predicted,
                "observed_direction": observation.direction,
                "direction_match": match, "slope": "%.6g" % slope,
                "reporting": observation.reporting, "source": observation.source,
            })
    out.append("")
    out.append("Direction agreement: %d of %d" % (correct, scored))
    out.append("")
    return rows


def magnitude_section(out, compounds, models, responses, rows):
    out.append(_rule("6. Magnitude: the error per compound per endpoint"))
    out.append(_wrap(
        "'residual_sd' is how far the NEAREST prediction the model can make "
        "sits from the observation, in units of the observation's own "
        "standard deviation. A zero means the model can be FITTED to the "
        "observation, not that it predicted it, and 'implied x' is the "
        "gain x activation product that fit would need."
    ))
    out.append("")
    header = ("%-12s %-11s %-7s %-9s %-9s %-9s %-9s %s"
              % ("compound", "endpoint", "dose", "observed", "obs sd",
                 "reach lo", "reach hi", "residual_sd"))
    out.append(header)
    out.append("-" * len(header))
    any_row = False
    for name, compound in sorted(compounds.items()):
        target = None
        for candidate in compound.modelled_targets:
            if hv.behavioural_path_status_for(compound, candidate) == "available":
                target = candidate
                break
        for observation in compound.behavioural_observations:
            if not observation.is_quantitative or target is None:
                continue
            any_row = True
            response = _response_for(target, models, responses)
            result = hv.magnitude_result(name, target, response, observation)
            out.append("%-12s %-11s %-7.3g %-9.4f %-9.4f %-9.4f %-9.4f %.3f (%s)" % (
                name, observation.endpoint, observation.dose_mg_per_kg,
                result.observed_ratio, result.observed_sd,
                result.reachable.ratio_lo, result.reachable.ratio_hi,
                result.residual_sd, result.verdict,
            ))
            for row in rows:
                if (row["compound"] == name
                        and row["endpoint"] == observation.endpoint
                        and row["dose_mg_per_kg"] == observation.dose_mg_per_kg):
                    row.update({
                        "observed_ratio": "%.6g" % result.observed_ratio,
                        "observed_sd": "%.6g" % result.observed_sd,
                        "reachable_lo": "%.6g" % result.reachable.ratio_lo,
                        "reachable_hi": "%.6g" % result.reachable.ratio_hi,
                        "residual_sd": "%.6g" % result.residual_sd,
                        "magnitude_verdict": result.verdict,
                        "implied_x": ("" if result.implied_x is None
                                      else "%.6g" % result.implied_x),
                    })
    if not any_row:
        out.append("  (no compound has both a quantitative observation and a "
                   "behavioural path)")
    out.append("")


def monotonicity_section(out, compounds):
    out.append(_rule("7. Monotonicity: the test no gain can pass"))
    out.append(_wrap(
        "Every stage from dose to endpoint index is monotone, so the model "
        "cannot produce a dose-response that rises and then falls, at any "
        "gain, at any target, with any density map. A published series that "
        "turns over refutes the model outright. Whether each turnover "
        "survives its own error bars is computed at two thresholds, because "
        "reporting only the threshold that passes would be choosing the test "
        "after seeing the data."
    ))
    out.append("")
    tested = 0
    for name, compound in sorted(compounds.items()):
        endpoints = sorted({o.endpoint for o in compound.behavioural_observations})
        for endpoint in endpoints:
            verdicts = [
                hv.monotonicity_verdict(name, endpoint,
                                        compound.behavioural_observations,
                                        n_sigma=sigma)
                for sigma in SIGMA_LEVELS
            ]
            if verdicts[0] is None:
                continue
            tested += 1
            base = verdicts[0]
            out.append("  %s / %s" % (name, endpoint))
            out.append("    doses  %s" % ", ".join("%g" % d for d in base.doses))
            out.append("    ratios %s" % ", ".join(
                "%.4f+/-%.4f" % (r, s) for r, s in zip(base.ratios, base.sds)
            ))
            out.append("    turns over: %s" % base.turns_over)
            for sigma, verdict in zip(SIGMA_LEVELS, verdicts):
                out.append("      at %.2f sd: %s" % (
                    sigma, "REFUTES" if verdict.significant
                    else "not resolved"))
            out.append(_wrap(base.note, width=68, indent="    "))
            out.append("")
    if not tested:
        out.append("  (no compound has three or more quantitative dose "
                   "levels at one endpoint)")
    out.append("")


def occupancy_section(out, compounds):
    out.append(_rule("8. The occupancy tier, and why it could not be run"))
    out.append(_wrap(
        "Occupancy is the one prediction with NO unfitted parameter in it: "
        "PK, the partition and the Ki fix it completely. It is therefore the "
        "tier whose absence costs most, and what blocks it is named per "
        "compound rather than summarised."
    ))
    out.append("")
    total = 0
    for name, compound in sorted(compounds.items()):
        gap = hv.occupancy_gap(compound)
        total += gap["observations"]
        if not gap["observations"]:
            continue
        out.append("  %s: %d published occupancy observation(s), blocked by %s"
                   % (name, gap["observations"], ", ".join(gap["blocked_by"])))
        for observation in compound.occupancy_observations:
            out.append("      %g mg/kg %s %s -> %.0f%% at %s [%s]" % (
                observation.dose_mg_per_kg, observation.route,
                observation.species, 100.0 * observation.occupancy,
                observation.target, observation.source.split("]")[0].strip("["),
            ))
        out.append("")
    out.append(_wrap(
        "%d occupancy observations were collected and NONE could be scored, "
        "because no held-out compound has both a dose-to-concentration model "
        "and a plasma-to-brain-unbound partition." % total
    ))
    out.append("")


def searches_section(out, compounds):
    out.append(_rule("9. Searched and empty"))
    searched = notsearched = 0
    for name, compound in sorted(compounds.items()):
        for record in compound.searches:
            if record.searched:
                searched += 1
            else:
                notsearched += 1
    out.append(_wrap(
        "%d documented searches that came back empty, and %d gaps that were "
        "NOT searched. The two are counted separately because presenting an "
        "unrun search as a negative result would be the most consequential "
        "overclaim available in this task."
        % (searched, notsearched)
    ))
    out.append("")
    for name, compound in sorted(compounds.items()):
        if not compound.searches:
            continue
        out.append("  %s" % name)
        for record in compound.searches:
            out.append(_wrap(
                "%s %s" % ("[SEARCHED]" if record.searched else "[NOT SEARCHED]",
                           record.what),
                width=68, indent="    ",
            ))
            out.append(_wrap(record.conclusion, width=66, indent="        "))
            out.append("")
    out.append("")


def write_figures(compounds, models, responses, ceiling, statuses):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- reachable ranges, with the observations on top --------------------
    targets = sorted(responses)
    if targets:
        fig, axes = plt.subplots(
            1, len(targets), figsize=(1 + 4.2 * len(targets), 5), squeeze=False
        )
        for ax, target in zip(axes[0], targets):
            ranges = hf.reachable_index_range(responses[target])
            keys = list(hf.ENDPOINT_KEYS)
            y = np.arange(len(keys))
            lo = [ranges[k].ratio_lo for k in keys]
            hi = [ranges[k].ratio_hi for k in keys]
            ax.barh(y, np.array(hi) - np.array(lo), left=lo, height=0.55,
                    color="#c7d9ec", edgecolor="#4a6f96")
            ax.axvline(1.0, color="0.35", lw=1, ls="--")
            for name, compound in sorted(compounds.items()):
                for obs in compound.behavioural_observations:
                    if not obs.is_quantitative or obs.endpoint not in keys:
                        continue
                    if target not in compound.modelled_targets:
                        continue
                    idx = keys.index(obs.endpoint)
                    ax.errorbar(obs.ratio, idx, xerr=obs.ratio_sd, fmt="o",
                                color="#b03030", ms=5, capsize=3, zorder=5)
            ax.set_yticks(y)
            ax.set_yticklabels(keys, fontsize=8)
            ax.set_xlabel("endpoint index, drug / vehicle")
            ax.set_title("%s (%s)" % (target, fp.TARGETS[target].gene),
                         fontsize=10)
            ax.invert_yaxis()
        fig.suptitle(
            "What the whole-brain model can reach over ALL gains (bars), "
            "against published effects (points)", fontsize=11
        )
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, "heldout_reachable.png"), dpi=150)
        plt.close(fig)

    # --- coverage matrix ---------------------------------------------------
    names = sorted(statuses)
    grid = np.array([[1.0 if statuses[n].links[l] else 0.0
                      for l in hv.CHAIN_LINKS] for n in names])
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(names) + 2.4))
    ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(hv.CHAIN_LINKS)))
    ax.set_xticklabels(hv.CHAIN_LINKS, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    for i in range(len(names)):
        for j in range(len(hv.CHAIN_LINKS)):
            ax.text(j, i, "yes" if grid[i, j] else "NO", ha="center",
                    va="center", fontsize=8,
                    color="0.15" if grid[i, j] else "white")
    ax.set_title("Held-out compounds: which links of the\n"
                 "dose-to-behaviour chain exist", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "heldout_coverage.png"), dpi=150)
    plt.close(fig)

    # --- the tone ceiling --------------------------------------------------
    items = sorted(ceiling.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.barh([k for k, _ in items], [v for _, v in items], color="#6b8fb4")
    ax.set_xscale("log")
    ax.set_xlabel("maximum |delta activation| a full antagonist can produce")
    ax.set_title("The tone ceiling: what a blocker can remove at each target\n"
                 "(set by P6.2's measured basal transmitter, not by affinity)",
                 fontsize=10)
    ax.axvline(0.01, color="0.4", ls="--", lw=1)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "heldout_tone_ceiling.png"), dpi=150)
    plt.close(fig)


def main():
    if not os.path.isdir(RESULTS):
        os.makedirs(RESULTS)
    compounds = ho.held_out_compounds()

    out = []
    out.append("Held-out drug validation (P8.2)")
    out.append("=" * 72)
    out.append(_wrap(
        "Predictions for %d compounds never used in any fitting step, from "
        "published binding and pharmacokinetics alone. The training set is "
        "%s, read off the evidence module rather than restated."
        % (len(compounds), ", ".join(sorted(ho.training_compounds())))
    ))
    overlap = sorted(ho.training_compounds() & set(compounds))
    out.append("Overlap between held-out and training sets: %s"
               % (", ".join(overlap) if overlap else "none"))
    out.append("")

    statuses = coverage_section(out, compounds)
    binding_section(out, compounds)
    ceiling = tone_section(out)

    wanted = sorted({
        target
        for compound in compounds.values()
        for target in compound.modelled_targets
        if hv.behavioural_path_status_for(compound, target) == "available"
    })
    models, responses = {}, {}
    for target in wanted:
        model, _ = hf.build_multi_target_model([target], "heldout-%s" % target)
        models[target] = model
    reachable_section(out, models, responses, wanted)
    rows = direction_section(out, compounds, models, responses)
    magnitude_section(out, compounds, models, responses, rows)
    monotonicity_section(out, compounds)
    occupancy_section(out, compounds)
    searches_section(out, compounds)

    with open(os.path.join(RESULTS, "heldout_validation.txt"), "w") as handle:
        handle.write("\n".join(out) + "\n")

    fields = ["compound", "endpoint", "dose_mg_per_kg", "via_target",
              "predicted_direction", "observed_direction", "direction_match",
              "slope", "observed_ratio", "observed_sd", "reachable_lo",
              "reachable_hi", "residual_sd", "magnitude_verdict", "implied_x",
              "reporting", "source"]
    with open(os.path.join(RESULTS, "heldout_errors.csv"), "w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(os.path.join(RESULTS, "heldout_coverage.csv"), "w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["compound", "link", "available", "detail"])
        for name, status in sorted(statuses.items()):
            for link in hv.CHAIN_LINKS:
                writer.writerow([name, link, status.links[link],
                                 status.detail[link]])

    write_figures(compounds, models, responses, ceiling, statuses)
    print("\n".join(out))


if __name__ == "__main__":
    main()
